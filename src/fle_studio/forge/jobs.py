"""Forge jobs: a recipe turned into steps that run in the background, with progress, a log, cancel and resume.

A job is a list of steps (importance matrix, build, router statistics, pruning, reference logits, KL divergence,
checksums). They run one after the other in one thread; one job runs at a time because every step is a large job
for the disk and the memory. The state of a job is a JSON file in its folder, saved after every change, so a job
that was cancelled, failed, or interrupted by closing Studio can be resumed: finished steps are not repeated, and a
step that writes a model continues at the last finished tensor.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import paths, procutil
from . import recipe as recipes
from . import tools as toolbox
from .ggufio import Cancelled

STEP_TITLES = {
    "imatrix": "Importance matrix",
    "build": "Build the model",
    "reap_table": "Router statistics",
    "reap_prune": "Prune experts",
    "kld_base": "Reference probabilities",
    "kld": "KL divergence",
    "finish": "Checksums and files",
}
LOG_KEEP = 4000
FINAL_STATES = ("done", "failed", "cancelled", "interrupted")


class StepError(RuntimeError):
    """A step failed for a reason the user can act on; the message is shown as it is."""


def plan_steps(r: dict[str, Any]) -> list[str]:
    steps: list[str] = []
    if r["mode"] == "quantize" and r["quantize"]["imatrix"] and not r["imatrix"]["file"]:
        steps.append("imatrix")
    steps.append("build")
    if r["reap"]["enabled"]:
        if not r["reap"]["table"]:
            steps.append("reap_table")
        steps.append("reap_prune")
    if r["verify"]["enabled"]:
        if not r["verify"]["base_logits"]:
            steps.append("kld_base")
        steps.append("kld")
    steps.append("finish")
    return steps


def needed_tools(r: dict[str, Any]) -> list[str]:
    out: list[str] = []
    steps = plan_steps(r)
    if "imatrix" in steps:
        out.append("llama-imatrix")
    if r["mode"] == "quantize":
        out.append("llama-quantize")
    if "kld" in steps:
        out.append("llama-perplexity")
    return out


class Job:
    def __init__(self, folder: Path, data: dict[str, Any]) -> None:
        self.folder = folder
        self.data = data
        self.log: deque[dict[str, Any]] = deque(maxlen=LOG_KEEP)
        self.seq = 0
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None
        self._saved = 0.0

    @property
    def id(self) -> str:
        return str(self.data["id"])

    @property
    def state(self) -> str:
        return str(self.data["state"])

    @property
    def work(self) -> Path:
        return paths.ensure(self.folder / "work")

    def save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._saved < 0.5:
            return
        self._saved = now
        self.data["updated"] = time.time()
        tmp = self.folder / "job.tmp"
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.folder / "job.json")


class Ctx:
    """What a step may do: log, report progress, run a tool, ask whether the job was cancelled."""

    def __init__(self, manager: ForgeJobs, job: Job, step: dict[str, Any]) -> None:
        self.manager, self.job, self.step = manager, job, step
        self.recipe: dict[str, Any] = job.data["recipe"]
        self.artifacts: dict[str, Any] = job.data["artifacts"]
        self.tools = manager.tools()
        self.out_dir = manager.out_dir()

    @property
    def work(self) -> Path:
        return self.job.work

    def log(self, text: str, kind: str = "") -> None:
        self.manager.add_log(self.job, text, kind)

    def progress(self, fraction: float, message: str = "") -> None:
        with self.manager.lock:
            self.step["progress"] = max(0.0, min(1.0, fraction))
            if message:
                self.step["message"] = message
        self.job.save()

    def cancelled(self) -> bool:
        return self.job.cancel.is_set()

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled

    def tool(self, name: str) -> toolbox.Tool:
        t = self.tools.get(name)
        if not t or not t.ok:
            raise StepError(
                f"{name} was not found. Set the folder of your llama.cpp tools on the Forge page, or "
                "install an engine on the Engine page (its folder is searched too)."
            )
        return t

    def run(self, tool: toolbox.Tool, args: list[str], on_line: Callable[[str], None] | None = None) -> int:
        """Run a tool, stream its output into the log, stop it when the job is cancelled. Returns the exit code."""
        cmd = [*toolbox.command_for(tool.path), *args]
        self.log("$ " + " ".join(_quote(c) for c in cmd), "cmdline")
        flags = procutil.popen_kwargs()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=toolbox.environment(tool.path),
            cwd=str(Path(tool.path).parent),
            **flags,
        )
        self.job.proc = proc
        done = threading.Event()

        def watch() -> None:
            while not done.wait(0.25):
                if self.job.cancel.is_set() and proc.poll() is None:
                    procutil.ask_to_stop(proc)
                    if not done.wait(10) and proc.poll() is None:
                        proc.kill()
                    return

        threading.Thread(target=watch, daemon=True).start()
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                low = line.lower()
                self.log(line, "err" if ("error" in low or "failed" in low) else "")
                if on_line:
                    on_line(line)
            code = proc.wait()
        finally:
            done.set()
            self.job.proc = None
            if proc.poll() is None:
                proc.kill()
        if self.cancelled():
            raise Cancelled
        return code


def _quote(word: str) -> str:
    return f'"{word}"' if " " in word else word


class ForgeJobs:
    def __init__(
        self,
        data_dir: Path,
        out_dir: Callable[[], Path],
        tools: Callable[[], dict[str, toolbox.Tool]],
        runners: dict[str, Callable[[Ctx], dict[str, Any]]] | None = None,
    ) -> None:
        self.dir = paths.ensure(data_dir / "forge" / "jobs")
        self._out_dir = out_dir
        self._tools = tools
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}
        if runners is None:
            from .steps import RUNNERS

            runners = RUNNERS
        self.runners = runners
        for folder in sorted(self.dir.iterdir()) if self.dir.exists() else []:
            try:
                data = json.loads((folder / "job.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            job = Job(folder, data)
            if job.state in ("queued", "running"):
                job.data["state"] = "interrupted"
                for s in job.data["steps"]:
                    if s["state"] == "running":
                        s["state"] = "interrupted"
                job.save(force=True)
            self._load_log(job)
            self.jobs[job.id] = job

    # ------------------------------------------------------------------------------------------ helpers

    def tools(self) -> dict[str, toolbox.Tool]:
        return self._tools()

    def out_dir(self) -> Path:
        return paths.ensure(self._out_dir())

    def _load_log(self, job: Job) -> None:
        try:
            lines = (job.folder / "log.txt").read_text(encoding="utf-8", errors="replace").splitlines()[-LOG_KEEP:]
        except OSError:
            return
        for text in lines:
            job.seq += 1
            job.log.append({"seq": job.seq, "t": 0.0, "text": text, "kind": ""})

    def add_log(self, job: Job, text: str, kind: str = "") -> None:
        with self.lock:
            job.seq += 1
            job.log.append({"seq": job.seq, "t": time.time(), "text": text, "kind": kind})
        try:
            with open(job.folder / "log.txt", "a", encoding="utf-8", errors="replace") as f:
                f.write(text + "\n")
        except OSError:
            pass

    def running(self) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs.values() if j.thread and j.thread.is_alive()), None)

    # ------------------------------------------------------------------------------------------ public API

    def check(self, r: dict[str, Any]) -> dict[str, Any]:
        """Normalise a recipe and say what would run and what is missing, without starting anything."""
        norm = recipes.normalize(r)
        problems = recipes.validate_sources(norm)
        for sid, s in norm["sources"].items():
            if s["path"] and not Path(s["path"]).expanduser().is_file():
                problems.append(f"source {sid}: no file at {s['path']}")
        tools = self.tools()
        missing = toolbox.missing_for(tools, tuple(needed_tools(norm)))
        feature_problems = []
        q = norm["quantize"]
        if norm["mode"] == "quantize" and "llama-quantize" in tools:
            f = tools["llama-quantize"].features
            if q["rules"] and not f.get("tensor_type", True):
                feature_problems.append("this llama-quantize has no --tensor-type; the per-tensor rules cannot be used")
            if q["allow_requantize"] and not f.get("allow_requantize", True):
                feature_problems.append("this llama-quantize has no --allow-requantize")
        if (
            "kld" in plan_steps(norm)
            and "llama-perplexity" in tools
            and not tools["llama-perplexity"].features.get("kl_divergence", True)
        ):
            feature_problems.append("this llama-perplexity has no --kl-divergence")
        steps = plan_steps(norm)

        def missing_text(path: str) -> bool:
            return not path or not Path(path).expanduser().is_file()

        corpus = norm["imatrix"]["corpus"]
        if "imatrix" in steps and missing_text(corpus):
            problems.append("The calibration text (imatrix.corpus) is not set or not a file")
        if "reap_table" in steps and "llama-fle-reap" in tools and tools["llama-fle-reap"].ok and missing_text(corpus):
            problems.append("Pruning measures the router on the calibration text (imatrix.corpus), which is not set")
        if "kld" in steps and missing_text(norm["verify"]["text"]):
            problems.append("The text for the comparison (verify.text) is not set or not a file")
        return {
            "recipe": norm,
            "steps": [{"id": s, "title": STEP_TITLES[s]} for s in steps],
            "problems": problems + feature_problems,
            "missing_tools": missing,
            "ok": not problems and not missing and not feature_problems,
        }

    def create(self, r: dict[str, Any], start: bool = True) -> dict[str, Any]:
        info = self.check(r)
        if not info["ok"]:
            raise ValueError("; ".join([*info["problems"], *[f"{t} is missing" for t in info["missing_tools"]]]))
        norm = info["recipe"]
        if self.running():
            raise RuntimeError("Another Forge job is running. Wait for it or cancel it first.")
        ident = secrets.token_hex(6)
        folder = paths.ensure(self.dir / ident)
        final = self._unique_output(norm["name"])
        data = {
            "id": ident,
            "name": norm["name"],
            "recipe": norm,
            "created": time.time(),
            "updated": time.time(),
            "state": "queued",
            "error": "",
            "artifacts": {"final": str(final)},
            "result": {},
            "steps": [
                {
                    "id": s["id"],
                    "title": s["title"],
                    "state": "pending",
                    "progress": 0.0,
                    "message": "",
                    "started": None,
                    "finished": None,
                    "error": "",
                    "outputs": {},
                }
                for s in info["steps"]
            ],
        }
        job = Job(folder, data)
        with self.lock:
            self.jobs[ident] = job
        job.save(force=True)
        self.add_log(
            job, f"Job {ident}: {norm['name']} ({norm['mode']}), steps: " + ", ".join(s["id"] for s in info["steps"])
        )
        if start:
            self._start(job)
        return self.public(job)

    def _unique_output(self, name: str) -> Path:
        base = self.out_dir()
        taken = {j.data["artifacts"].get("final") for j in self.jobs.values()}
        n, candidate = 1, base / f"{name}.gguf"
        while candidate.exists() or str(candidate) in taken or candidate.with_name(candidate.name + ".part").exists():
            n += 1
            candidate = base / f"{name}-{n}.gguf"
        return candidate

    def _start(self, job: Job) -> None:
        job.cancel.clear()
        job.data["state"] = "running"
        job.data["error"] = ""
        job.save(force=True)
        job.thread = threading.Thread(target=self._run, args=(job,), name=f"forge-{job.id}", daemon=True)
        job.thread.start()

    def resume(self, ident: str) -> dict[str, Any]:
        job = self.get_job(ident)
        if job.state not in ("failed", "cancelled", "interrupted"):
            raise RuntimeError(f"A job that is {job.state} cannot be resumed")
        if self.running():
            raise RuntimeError("Another Forge job is running")
        with self.lock:
            for s in job.data["steps"]:
                if s["state"] != "done":
                    s.update({"state": "pending", "error": "", "progress": 0.0, "message": ""})
        self.add_log(job, "Resuming")
        self._start(job)
        return self.public(job)

    def cancel(self, ident: str) -> dict[str, Any]:
        job = self.get_job(ident)
        job.cancel.set()
        if job.proc is not None and job.proc.poll() is None:
            procutil.ask_to_stop(job.proc)
        return self.public(job)

    def delete(self, ident: str, keep_model: bool = True) -> None:
        job = self.get_job(ident)
        if job.thread and job.thread.is_alive():
            raise RuntimeError("Cancel the job before deleting it")
        shutil.rmtree(job.folder, ignore_errors=True)
        with self.lock:
            self.jobs.pop(ident, None)
        if not keep_model:
            final = Path(job.data["artifacts"].get("final", ""))
            for p in (final, final.with_name(final.name + ".part"), final.with_name(final.name + ".part.json")):
                if p.name and p.is_file():
                    p.unlink()

    def get_job(self, ident: str) -> Job:
        with self.lock:
            job = self.jobs.get(ident)
        if not job:
            raise KeyError(ident)
        return job

    def public(self, job: Job) -> dict[str, Any]:
        with self.lock:
            d = json.loads(json.dumps(job.data))
        d["log_seq"] = job.seq
        d["running"] = bool(job.thread and job.thread.is_alive())
        return d

    def get(self, ident: str) -> dict[str, Any]:
        return self.public(self.get_job(ident))

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda j: -j.data["created"])
        out = []
        for j in jobs:
            d = self.public(j)
            d.pop("recipe", None)
            out.append(d)
        return out

    def log_since(self, ident: str, since: int) -> dict[str, Any]:
        job = self.get_job(ident)
        with self.lock:
            lines = [dict(x) for x in job.log if x["seq"] > since]
        return {"lines": lines, "next": job.seq}

    # ------------------------------------------------------------------------------------------ the runner

    def _run(self, job: Job) -> None:
        data = job.data
        try:
            for step in data["steps"]:
                if step["state"] == "done":
                    continue
                if job.cancel.is_set():
                    raise Cancelled
                ctx = Ctx(self, job, step)
                with self.lock:
                    step.update(
                        {"state": "running", "started": time.time(), "progress": 0.0, "error": "", "message": ""}
                    )
                job.save(force=True)
                self.add_log(job, f"== {step['title']}", "head")
                try:
                    outputs = self.runners[step["id"]](ctx) or {}
                except Cancelled:
                    with self.lock:
                        step["state"] = "cancelled"
                    raise
                except Exception as exc:
                    with self.lock:
                        step.update({"state": "failed", "error": str(exc)[:600], "finished": time.time()})
                    if not isinstance(exc, StepError):
                        self.add_log(job, traceback.format_exc(), "err")
                    raise
                with self.lock:
                    step.update({"state": "done", "progress": 1.0, "finished": time.time(), "outputs": outputs})
                    data["artifacts"].update({k: v for k, v in outputs.items() if isinstance(v, (str, int, float))})
                job.save(force=True)
            with self.lock:
                data["state"] = "done"
            self.add_log(job, "Done", "ok")
        except Cancelled:
            with self.lock:
                data["state"] = "cancelled"
            self.add_log(job, "Cancelled. Resume continues where it stopped.", "warn")
        except Exception as exc:
            with self.lock:
                data["state"], data["error"] = "failed", str(exc)[:600]
            self.add_log(job, f"Failed: {exc}", "err")
        finally:
            job.save(force=True)
