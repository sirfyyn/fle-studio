"""A small, repeatable speed test against the running engine: the 'experiments' button.

Three requests per run: a short answer, a longer piece of code, and a long prompt (about 6,000 tokens) that measures
prefill. Each run records prefill and decode speed, time to first token and MTP acceptance from the server's own
timings, with the launch profile and the engine, so runs with different settings can be compared side by side.
These are measurements on this machine, nothing more; with sampling on, decode speed and MTP acceptance vary from
run to run, so compare medians of several runs.
"""

from __future__ import annotations

import json
import secrets
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import paths, stats
from .engine import Engine


def _long_prompt() -> str:
    rows = []
    for i in range(1, 241):
        level = (i * 37) % 389 + 120
        temp = 4 + (i * 13) % 17
        rows.append(
            f"Station {i:03d} on the river records a water level of {level} centimetres, a water "
            f"temperature of {temp} degrees and {i % 7} boats passing before noon."
        )
    return (
        "Here is a log of measuring stations.\n\n"
        + "\n".join(rows)
        + "\n\nWhich station recorded the highest water level? Answer with the station number and the level."
    )


TESTS = (
    {
        "name": "Short answer",
        "max_tokens": 256,
        "prompt": "Explain in a few sentences why the sky is blue, for a curious ten-year-old.",
    },
    {
        "name": "Code",
        "max_tokens": 768,
        "prompt": "Write a Python function that parses an ISO 8601 duration like P3DT4H12M into seconds, with a "
        "docstring and three doctest examples. Then explain the edge cases.",
    },
    {"name": "Long prompt", "max_tokens": 128, "prompt": _long_prompt()},
)
GEN_TESTS = ("Short answer", "Code")  # the requests that measure generation speed
PREFILL_TESTS = ("Long prompt",)
# What the tuner and the A/B/A comparison run per measurement: no 768-token code answer, which takes the most time.
QUICK_TESTS = tuple(t for t in TESTS if t["name"] != "Code")


class Bench:
    def __init__(self, engine: Engine, data_dir: Path) -> None:
        self.engine = engine
        self.dir = paths.ensure(data_dir / "bench")
        self._lock = threading.Lock()
        self.current: dict[str, Any] | None = None
        self._cancel = threading.Event()

    def results(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.dir.glob("*.json"), reverse=True)[:200]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def delete(self, ident: str) -> None:
        for p in self.dir.glob(f"*-{ident}.json"):
            p.unlink()

    def start(
        self,
        *,
        runs: int,
        label: str,
        context: dict[str, Any],
        probe: Callable[[str, dict[str, str], Callable[[str], None]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Start the speed test in a thread. ``probe(base, headers, progress)`` runs after the speed requests and
        returns the quality result (the knowledge probe) that is stored in the same record, next to the speed."""
        with self._lock:
            if self.current and self.current["state"] == "running":
                raise RuntimeError("A speed test is already running")
            state, base, headers = self.engine.target()[:3]
            if state not in ("ready", "attached") or not base:
                raise RuntimeError("Load a model first")
            runs = max(1, min(10, int(runs)))
            self._cancel.clear()
            self.current = {
                "id": secrets.token_hex(4),
                "state": "running",
                "label": label.strip()[:80],
                "started": time.time(),
                "runs": runs,
                "done": 0,
                "total": runs * len(TESTS),
                "step": "",
                "tests": [],
                "context": context,
            }
            cur = self.current
        threading.Thread(target=self._run, args=(cur, base, headers, probe), daemon=True).start()
        return dict(cur)

    def cancel(self) -> None:
        self._cancel.set()

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self.current)) if self.current else None

    def _run(
        self,
        cur: dict[str, Any],
        base: str,
        headers: dict[str, str],
        probe: Callable[[str, dict[str, str], Callable[[str], None]], dict[str, Any]] | None = None,
    ) -> None:
        def step(run: int, test: dict[str, Any]) -> None:
            with self._lock:
                cur["step"] = f"Run {run + 1}/{cur['runs']}: {test['name']}"

        def result(res: dict[str, Any]) -> None:
            with self._lock:
                cur["tests"].append(res)
                cur["done"] += 1

        def say(text: str) -> None:
            if self._cancel.is_set():  # the probe reports every question, so Cancel reaches it
                raise InterruptedError
            with self._lock:
                cur["step"] = text

        try:
            self.measure(base, headers, cur["runs"], TESTS, step, result)
            if probe is not None:
                if self._cancel.is_set():
                    raise InterruptedError
                say("Knowledge probe")
                quality = probe(base, headers, say)
                with self._lock:
                    cur["quality"] = quality
            with self._lock:
                cur["state"] = "done"
        except InterruptedError:
            with self._lock:
                cur["state"] = "cancelled"
        except Exception as exc:  # report, never crash the thread silently
            with self._lock:
                cur["state"], cur["error"] = "failed", str(exc)
        with self._lock:
            cur["finished"] = time.time()
            cur["summary"] = self._summary(cur["tests"])
            cur["step"] = ""
            record = json.loads(json.dumps(cur))
        if record["tests"]:
            name = time.strftime("%Y%m%d-%H%M%S", time.localtime(record["started"])) + f"-{record['id']}.json"
            (self.dir / name).write_text(json.dumps(record, indent=1), encoding="utf-8")

    def _one(
        self,
        base: str,
        headers: dict[str, str],
        test: dict[str, Any],
        seed: int,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        cancel = cancel or self._cancel
        body = {
            "messages": [{"role": "user", "content": test["prompt"]}],
            "max_tokens": test["max_tokens"],
            "stream": True,
            "seed": seed,
            "cache_prompt": False,
            "timings_per_token": False,
        }
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        t0 = time.perf_counter()
        first = None
        timings: dict[str, Any] = {}
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if cancel.is_set():
                    raise InterruptedError
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.perf_counter()
                if chunk.get("timings"):
                    timings = chunk["timings"]
        wall = time.perf_counter() - t0
        draft_n = timings.get("draft_n") or 0
        return {
            "prompt_n": timings.get("prompt_n"),
            "prefill": timings.get("prompt_per_second"),
            "generated": timings.get("predicted_n"),
            "decode": timings.get("predicted_per_second"),
            "ttft_ms": round((first - t0) * 1000) if first else None,
            "accept": round(timings.get("draft_n_accepted", 0) / draft_n, 4) if draft_n else None,
            "wall_s": round(wall, 2),
        }

    def measure(
        self,
        base: str,
        headers: dict[str, str],
        runs: int,
        tests: tuple[dict[str, Any], ...] = TESTS,
        on_step: Callable[[int, dict[str, Any]], None] | None = None,
        on_result: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
        run_offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Run ``tests`` ``runs`` times against the engine at ``base``, one after the other, and return the results.

        Blocking; raises InterruptedError when ``cancel`` (default: this speed test's Cancel button) is set and the
        usual urllib errors when the engine fails. Used by the speed test thread here and by the tuner and the A/B/A
        comparison, which restart the engine between their measurements."""
        cancel = cancel or self._cancel
        out: list[dict[str, Any]] = []
        for run in range(runs):
            for i, test in enumerate(tests):
                if cancel.is_set():
                    raise InterruptedError
                if on_step:
                    on_step(run, test)
                res = self._one(base, headers, test, seed=1000 * (run + run_offset) + i, cancel=cancel)
                res.update({"run": run + run_offset + 1, "name": test["name"]})
                out.append(res)
                if on_result:
                    on_result(res)
        return out

    @staticmethod
    def _summary(tests: list[dict[str, Any]]) -> dict[str, Any]:
        def med(key: str, names: tuple[str, ...]) -> float | None:
            vals = [t[key] for t in tests if t.get(key) is not None and t["name"] in names]
            return round(statistics.median(vals), 2) if vals else None

        gen_tests = ("Short answer", "Code")
        return {
            "decode": med("decode", gen_tests),
            "prefill": med("prefill", ("Long prompt",)),
            "ttft_ms": med("ttft_ms", gen_tests),
            "accept": med("accept", gen_tests),
            "n": len(tests),
            **run_stats(tests),
        }


def per_run(tests: list[dict[str, Any]], key: str, names: tuple[str, ...]) -> list[float]:
    """One value per run: the median of ``key`` over the requests of that run that are named in ``names``."""
    by_run: dict[int, list[float]] = {}
    for t in tests:
        if t.get(key) is not None and t.get("name") in names:
            by_run.setdefault(int(t.get("run") or 0), []).append(float(t[key]))
    return [round(statistics.median(v), 4) for _, v in sorted(by_run.items())]


def run_stats(tests: list[dict[str, Any]]) -> dict[str, Any]:
    """The repeated runs of a result: one decode and one prefill speed per run, with the median and 95 % interval.

    ``decode_runs``/``prefill_runs`` are what the comparisons (stats.compare, A/B/A) work with; the interval needs at
    least 3 runs (``decode_ci.lo`` is None before)."""
    decode = per_run(tests, "decode", GEN_TESTS)
    prefill = per_run(tests, "prefill", PREFILL_TESTS)
    return {
        "decode_runs": decode,
        "prefill_runs": prefill,
        "decode_ci": stats.summarize(decode),
        "prefill_ci": stats.summarize(prefill),
    }


def check_reachable(base: str, headers: dict[str, str]) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/health", headers=headers), timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False
