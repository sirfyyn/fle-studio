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
from pathlib import Path
from typing import Any

from . import paths
from .engine import Engine


def _long_prompt() -> str:
    rows = []
    for i in range(1, 241):
        level = (i * 37) % 389 + 120
        temp = 4 + (i * 13) % 17
        rows.append(f"Station {i:03d} on the river records a water level of {level} centimetres, a water "
                    f"temperature of {temp} degrees and {i % 7} boats passing before noon.")
    return ("Here is a log of measuring stations.\n\n" + "\n".join(rows) +
            "\n\nWhich station recorded the highest water level? Answer with the station number and the level.")


TESTS = (
    {"name": "Short answer", "max_tokens": 256,
     "prompt": "Explain in a few sentences why the sky is blue, for a curious ten-year-old."},
    {"name": "Code", "max_tokens": 768,
     "prompt": "Write a Python function that parses an ISO 8601 duration like P3DT4H12M into seconds, with a "
               "docstring and three doctest examples. Then explain the edge cases."},
    {"name": "Long prompt", "max_tokens": 128, "prompt": _long_prompt()},
)


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

    def start(self, *, runs: int, label: str, context: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.current and self.current["state"] == "running":
                raise RuntimeError("A speed test is already running")
            state, base, headers = self.engine.target()[:3]
            if state not in ("ready", "attached") or not base:
                raise RuntimeError("Load a model first")
            runs = max(1, min(10, int(runs)))
            self._cancel.clear()
            self.current = {"id": secrets.token_hex(4), "state": "running", "label": label.strip()[:80],
                            "started": time.time(), "runs": runs, "done": 0, "total": runs * len(TESTS),
                            "step": "", "tests": [], "context": context}
            cur = self.current
        threading.Thread(target=self._run, args=(cur, base, headers), daemon=True).start()
        return dict(cur)

    def cancel(self) -> None:
        self._cancel.set()

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self.current)) if self.current else None

    def _run(self, cur: dict[str, Any], base: str, headers: dict[str, str]) -> None:
        try:
            for run in range(cur["runs"]):
                for i, test in enumerate(TESTS):
                    if self._cancel.is_set():
                        raise InterruptedError
                    with self._lock:
                        cur["step"] = f"Run {run + 1}/{cur['runs']}: {test['name']}"
                    res = self._one(base, headers, test, seed=1000 * run + i)
                    res.update({"run": run + 1, "name": test["name"]})
                    with self._lock:
                        cur["tests"].append(res)
                        cur["done"] += 1
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

    def _one(self, base: str, headers: dict[str, str], test: dict[str, Any], seed: int) -> dict[str, Any]:
        body = {
            "messages": [{"role": "user", "content": test["prompt"]}],
            "max_tokens": test["max_tokens"],
            "stream": True,
            "seed": seed,
            "cache_prompt": False,
            "timings_per_token": False,
        }
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers}, method="POST")
        t0 = time.perf_counter()
        first = None
        timings: dict[str, Any] = {}
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                if self._cancel.is_set():
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
        }


def check_reachable(base: str, headers: dict[str, str]) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/health", headers=headers), timeout=2) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False
