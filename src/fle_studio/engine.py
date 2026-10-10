"""The engine: one llama-server process that FLE Studio starts and watches, or a server it attaches to.

All process starts happen on one long-lived supervisor thread. On Linux the engine is told to exit when the thread
that started it ends (PR_SET_PDEATHSIG), so that thread must not be a short-lived HTTP request thread.
"""

from __future__ import annotations

import collections
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from . import gguf, paths, procutil
from .launch import Plan

# Log lines that mark a loading phase (checked in order, first match wins).
PHASES = (
    ("listening on", "Ready"),  # "srv main: listening on http://…"; older builds: "server is listening on …"
    ("model loaded", "Starting the server"),
    ("warming up", "Warming up"),
    ("moe expert cache enabled", "Setting up the GPU expert cache"),
    ("llama_kv_cache", "Allocating the KV cache"),
    ("llama_context:", "Creating the context"),
    ("load_tensors:", "Loading weights"),
    ("llama_params_fit", "Fitting the model into GPU memory"),
    ("llama_model_loader: loaded meta data", "Reading the model"),
)

# Error lines and what to tell the user about them.
HINTS = (
    (re.compile(r"out of memory|cudaMalloc failed|failed to allocate .*buffer|ErrorOutOfDeviceMemory", re.I),
     "The GPU ran out of memory. Lower the context or the GPU expert cache on the Engine page."),
    (re.compile(r"error while loading shared libraries|cannot open shared object|\.dll was not found|"
                r"Library not loaded|image not found", re.I),
     "The engine misses a library. Reinstall the engine on the Engine page."),
    (re.compile(r"CUDA driver version is insufficient|no CUDA-capable device|cudaGetDeviceCount", re.I),
     "CUDA could not start. Update the NVIDIA driver, or use the Vulkan or CPU engine."),
    (re.compile(r"failed to load model|error loading model|unable to load model", re.I),
     "The engine could not load the model. Check that the download finished (Models page)."),
    (re.compile(r"couldn't bind|address already in use|bind failed|only one usage of each socket address", re.I),
     "The engine port is taken. Set the port to 0 (automatic) on the Engine page."),
    (re.compile(r"error: (invalid|unknown) argument|unknown option|invalid value", re.I),
     "The engine does not know one of the arguments. Is this profile meant for another engine?"),
)

CACHE_RE = re.compile(r"moe-cache:.*?(?:steps=(\d+).*?)?hits=(\d+).*?misses=(\d+).*?hit-rate=([\d.]+)\s*%")
CACHE_ON_RE = re.compile(r"MoE expert cache enabled[:,]?\s*(.*)", re.I)
ACCEPT_RE = re.compile(r"draft acceptance(?: rate)?\s*=\s*([\d.]+)")


class EngineBusy(RuntimeError):
    pass


class Engine:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.log_dir = paths.ensure(data_dir / "logs")
        self._lock = threading.RLock()
        self._ops: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._job = procutil.KillOnCloseJob()
        self.proc: subprocess.Popen[bytes] | None = None
        self.plan: Plan | None = None
        self.state = "stopped"  # stopped | starting | ready | stopping | crashed | attached
        self.base_url = ""
        self.api_key = ""
        self.reachable = False
        self.lines: collections.deque[tuple[int, float, str]] = collections.deque(maxlen=5000)
        self.seq = 0
        self.props: dict[str, Any] = {}
        self.phase = ""
        self.error = ""
        self.hint = ""
        self.exit_code: int | None = None
        self.started = 0.0
        self.ready_at = 0.0
        self.load_total = 0
        self.load_read = 0
        self.rss: int | None = None
        self.cache: dict[str, Any] = {}
        self.accept_logged: float | None = None
        self.restart_on_crash = False
        self._crashes: collections.deque[float] = collections.deque(maxlen=10)
        self._stop_requested = False
        self._log_file: Any = None
        threading.Thread(target=self._supervise, name="engine-supervisor", daemon=True).start()

    # ------------------------------------------------------------------------------------------ control

    def start(self, plan: Plan) -> None:
        with self._lock:
            if self.state in ("starting", "ready", "stopping"):
                raise EngineBusy("An engine is already running. Stop it first.")
            self.reachable = False
            self.state, self.plan, self.phase = "starting", plan, "Starting the engine"
            self.error = self.hint = ""
            self.exit_code = None
            self.cache, self.accept_logged, self.props = {}, None, {}
            self._stop_requested = False
            self.base_url = f"http://127.0.0.1:{plan.port}"
            self.api_key = ""
        self._ops.put(("start", plan))

    def stop(self) -> None:
        with self._lock:
            if self.state == "attached":
                self._detach_locked()
                return
            if self.proc is None:
                if self.state != "crashed":
                    self.state = "stopped"
                return
            self._stop_requested = True
            self.state, self.phase = "stopping", "Stopping"
        self._ops.put(("stop", None))

    def attach(self, url: str, key: str = "") -> dict[str, Any]:
        url = url.strip().rstrip("/")
        if not url:
            raise ValueError("Enter the server address, e.g. http://127.0.0.1:8080")
        if not re.match(r"^https?://", url):
            url = "http://" + url
        url = re.sub(r"/v1$", "", url)
        with self._lock:
            if self.proc is not None:
                raise EngineBusy("Stop the engine FLE Studio started before attaching to another server.")
            self.state, self.base_url, self.api_key, self.plan = "attached", url, key, None
            self.error = self.hint = self.phase = ""
            self.cache, self.accept_logged, self.props = {}, None, {}
        self.refresh_attached()
        return self.status()

    def refresh_attached(self) -> None:
        code = self._health()
        props = self.get_json("/props") if code == 200 else None
        with self._lock:
            if self.state == "attached":
                self.reachable = code == 200
                if props:
                    self.props = props
                self.error = "" if code == 200 else ("The server is loading its model" if code == 503 else
                                                     "The server does not answer")

    def _detach_locked(self) -> None:
        self.state, self.base_url, self.api_key, self.reachable, self.props = "stopped", "", "", False, {}

    def shutdown(self) -> None:
        """Stop a running engine and wait for it; used when FLE Studio exits."""
        with self._lock:
            proc = self.proc
            self._stop_requested = True
        if proc and proc.poll() is None:
            procutil.ask_to_stop(proc)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ------------------------------------------------------------------------------------------ supervisor

    def _supervise(self) -> None:
        while True:
            op, arg = self._ops.get()
            try:
                if op == "start":
                    self._spawn(arg)
                elif op == "stop":
                    self._terminate()
            except Exception as exc:  # the supervisor must survive anything a start can throw
                with self._lock:
                    self.proc = None
                    self.state, self.error, self.phase = "crashed", f"Could not start the engine: {exc}", ""
                self._append(f"FLE Studio: {self.error}")

    def _spawn(self, plan: Plan) -> None:
        with self._lock:
            if self.state != "starting" or self.plan is not plan:  # stopped again before we got here
                return
        binary = Path(plan.binary)
        if not binary.is_file():
            raise FileNotFoundError(f"no engine at {binary}")
        env = os.environ.copy()
        env.update(plan.env)
        if paths.IS_LINUX:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in (str(binary.parent), env.get("LD_LIBRARY_PATH")) if p)
        elif paths.IS_WIN:
            env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
        cmd = [str(binary), *plan.args]
        if binary.suffix == ".py":  # development and test engines
            cmd = [sys.executable, str(binary), *plan.args]
        self._open_log()
        self._append("$ " + plan.command_line().replace("\n", " "))
        total = 0
        for part in [*gguf.split_parts(Path(plan.model.get("path", ""))), *([Path(plan.draft)] if plan.draft else [])]:
            try:
                total += part.stat().st_size
            except OSError:
                pass
        proc = subprocess.Popen(cmd, cwd=str(binary.parent), env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **procutil.popen_kwargs())
        self._job.add(proc)
        with self._lock:
            self.proc, self.started, self.ready_at = proc, time.time(), 0.0
            self.load_total, self.load_read, self.rss = total, 0, None
        threading.Thread(target=self._read_output, args=(proc,), name="engine-log", daemon=True).start()
        threading.Thread(target=self._wait_ready, args=(proc,), name="engine-ready", daemon=True).start()

    def _terminate(self) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        procutil.ask_to_stop(proc)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self._append("FLE Studio: the engine did not stop within 20 s; killing it")
            proc.kill()
            proc.wait(timeout=10)

    def _read_output(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.strip():
                self._on_line(line)
        rc = proc.wait()
        self._on_exit(proc, rc)

    def _wait_ready(self, proc: subprocess.Popen[bytes]) -> None:
        while proc.poll() is None:
            with self._lock:
                if self.proc is not proc or self.state != "starting":
                    return
            read, rss = procutil.read_and_memory(proc)
            with self._lock:
                if read is not None:
                    self.load_read = read
                self.rss = rss
            if self._health() == 200:
                props = self.get_json("/props") or {}
                with self._lock:
                    if self.proc is proc and self.state == "starting":
                        self.state, self.phase, self.ready_at = "ready", "Ready", time.time()
                        self.props, self.reachable = props, True
                return
            time.sleep(0.5)

    def _on_exit(self, proc: subprocess.Popen[bytes], rc: int) -> None:
        again: Plan | None = None
        with self._lock:
            if self.proc is not proc:
                return
            self.proc, self.exit_code, self.reachable = None, rc, False
            if self._stop_requested:
                self.state, self.phase = "stopped", ""
            else:
                self.state, self.phase = "crashed", ""
                if not self.error:
                    last = next((t for _, _, t in reversed(self.lines) if not t.startswith("FLE Studio:")), "")
                    self.error = f"The engine exited (code {rc})" + (f": {last}" if last else "")
                now = time.time()
                self._crashes.append(now)
                recent = sum(1 for t in self._crashes if now - t < 600)
                if self.restart_on_crash and self.plan and self.ready_at and recent <= 3:
                    again = self.plan
        self._append(f"FLE Studio: engine exited with code {rc}")
        self._close_log()
        if again:
            self._append("FLE Studio: restart on crash is on; starting the engine again in 5 s")
            plan = again

            def restart() -> None:
                try:
                    self.start(plan)
                except EngineBusy:
                    pass

            threading.Timer(5.0, restart).start()

    # ------------------------------------------------------------------------------------------ log

    def _open_log(self) -> None:
        self._close_log()
        logs = sorted(self.log_dir.glob("engine-*.log"))
        for old in logs[:-9]:  # keep the last ten
            try:
                old.unlink()
            except OSError:
                pass
        try:
            self._log_file = (self.log_dir / time.strftime("engine-%Y%m%d-%H%M%S.log")).open(
                "a", encoding="utf-8", errors="replace")
        except OSError:
            self._log_file = None

    def _close_log(self) -> None:
        with self._lock:
            f, self._log_file = self._log_file, None
        if f:
            try:
                f.close()
            except OSError:
                pass

    def _append(self, text: str) -> None:
        with self._lock:
            self.seq += 1
            self.lines.append((self.seq, time.time(), text))
            f = self._log_file
        if f:
            try:
                f.write(text + "\n")
                f.flush()
            except (OSError, ValueError):
                pass

    def _on_line(self, line: str) -> None:
        self._append(line)
        low = line.lower()
        with self._lock:
            if self.state == "starting":
                for needle, phase in PHASES:
                    if needle in low:
                        self.phase = phase
                        break
            m = CACHE_RE.search(line)
            if m:
                self.cache.update({"steps": int(m.group(1) or 0), "hits": int(m.group(2)), "misses": int(m.group(3)),
                                   "hit_rate": float(m.group(4)), "at": time.time()})
            else:
                m = CACHE_ON_RE.search(line)
                if m:
                    self.cache["enabled"] = m.group(1).strip() or "on"
            m = ACCEPT_RE.search(line)
            if m:
                self.accept_logged = float(m.group(1))
            if not self.hint:
                for rx, hint in HINTS:
                    if rx.search(line):
                        self.hint, self.error = hint, line.strip()[:400]
                        break

    def log_since(self, seq: int, limit: int = 2000) -> dict[str, Any]:
        with self._lock:
            out = [{"seq": s, "t": t, "text": x} for s, t, x in self.lines if s > seq]
        return {"lines": out[-limit:], "seq": self.seq}

    # ------------------------------------------------------------------------------------------ HTTP

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _health(self) -> int:
        try:
            req = urllib.request.Request(self.base_url + "/health", headers=self.headers())
            with urllib.request.urlopen(req, timeout=2) as r:
                return int(r.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)
        except (OSError, ValueError):
            return 0

    def get_json(self, path: str, timeout: float = 3.0) -> Any:
        if not self.base_url:
            return None
        try:
            req = urllib.request.Request(self.base_url + path, headers=self.headers())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (OSError, ValueError):
            return None

    # ------------------------------------------------------------------------------------------ status

    def status(self) -> dict[str, Any]:
        with self._lock:
            plan = self.plan
            starting = self.state == "starting"
            progress = 1.0 if self.state in ("ready", "attached") else 0.0
            if starting and self.load_total:
                progress = min(0.99, self.load_read / self.load_total)
            props = self.props or {}
            dgs = props.get("default_generation_settings") or {}
            model = dict(plan.model) if plan else {}
            if not model and props:
                path = props.get("model_path") or ""
                model = {"name": props.get("model_alias") or Path(path).name or "model", "path": path}
            return {
                "state": self.state,
                "phase": self.phase,
                "error": self.error,
                "hint": self.hint,
                "exit_code": self.exit_code,
                "reachable": self.reachable,
                "attached": self.state == "attached",
                "base_url": self.base_url if self.state in ("starting", "ready", "attached") else "",
                "pid": self.proc.pid if self.proc else None,
                "started": self.started,
                "ready_at": self.ready_at,
                "load": {"read": self.load_read, "total": self.load_total, "progress": progress,
                         "elapsed": (time.time() - self.started) if starting else None},
                "plan": plan.public() if plan else None,
                "model": {k: model.get(k) for k in ("id", "name", "path", "catalog_id", "size")},
                "n_ctx": dgs.get("n_ctx") or props.get("n_ctx"),
                "slots": props.get("total_slots"),
                "build": props.get("build_info", ""),
                "cache": dict(self.cache),
                "accept_logged": self.accept_logged,
                "rss": self.rss,
                "log_seq": self.seq,
            }

    def target(self) -> tuple[str, str, dict[str, str], subprocess.Popen[bytes] | None]:
        """(state, base url, headers, process) for the samplers and the chat proxy."""
        with self._lock:
            return self.state, self.base_url, self.headers(), self.proc

    def note_memory(self, rss: int | None) -> None:
        with self._lock:
            self.rss = rss
