"""Live inference data, sampled about once a second.

Speeds come from the counters in llama-server's /metrics: tokens generated per second of generation time, prompt
tokens per second of prompt time, and the share of drafted (MTP) tokens that were accepted, each over the last
interval. A server started without --metrics has no such counters; then the timings of the last answer stand in.
The FLE expert-cache hit rate comes from the engine log (the engine prints it every 512 decode steps).
"""

from __future__ import annotations

import collections
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from . import procutil, system
from .engine import Engine

P = "llamacpp:"


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        name = name.split("{", 1)[0].strip()
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


class Live:
    def __init__(self, engine: Engine, interval: float = 1.0) -> None:
        self.engine = engine
        self.interval = interval
        self.history: collections.deque[dict[str, Any]] = collections.deque(maxlen=900)
        self.responses: collections.deque[dict[str, Any]] = collections.deque(maxlen=100)
        self.metrics_available: bool | None = None
        self._prev: dict[str, float] | None = None
        self._prev_url = ""
        self._lock = threading.Lock()
        self._gpus: list[dict[str, Any]] = []
        self._has_nvidia = bool(system.nvidia())
        self.totals: dict[str, float] = {}
        threading.Thread(target=self._loop, name="live-sampler", daemon=True).start()

    def record_response(self, timings: dict[str, Any], usage: dict[str, Any] | None = None) -> None:
        """Called by the chat proxy with the `timings` of every finished answer."""
        entry = {"t": time.time(), **{k: timings.get(k) for k in (
            "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
            "draft_n", "draft_n_accepted", "cache_n")}}
        if usage:
            entry["usage"] = usage
        with self._lock:
            self.responses.append(entry)

    def _loop(self) -> None:
        tick = 0
        while True:
            t0 = time.time()
            try:
                self._sample(tick)
            except Exception:  # a sampler hiccup must never stop the sampler
                pass
            tick += 1
            time.sleep(max(0.2, self.interval - (time.time() - t0)))

    def _get(self, base: str, path: str, headers: dict[str, str]) -> tuple[int, str]:
        try:
            req = urllib.request.Request(base + path, headers=headers)
            with urllib.request.urlopen(req, timeout=2) as r:
                return int(r.status), r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return int(exc.code), ""
        except (OSError, ValueError):
            return 0, ""

    def _sample(self, tick: int) -> None:
        bound = self.engine.bound()
        if bound == "service" or (bound == "url" and tick % 5 == 0):
            self.engine.refresh_attached()  # a service is read every second: it can start or stop at any time
        state, base, headers, proc = self.engine.target()
        s: dict[str, Any] = {"t": round(time.time(), 2), "state": state}
        mem = system.memory()
        s["ram_used"], s["ram_total"] = mem["total"] - mem["available"], mem["total"]
        if self._has_nvidia and (tick % 2 == 0 or not self._gpus):
            self._gpus = [{k: g.get(k) for k in ("index", "name", "utilization", "memory_used", "memory_total",
                                                 "power_w", "temperature_c")} for g in system.nvidia()]
        s["gpus"] = self._gpus
        if proc is not None:
            rss = procutil.read_and_memory(proc)[1]
            s["engine_mem"] = rss
            self.engine.note_memory(rss)
        elif bound == "service":
            s["engine_mem"] = self.engine.rss
        live = state in ("ready", "attached")
        if live and base:
            if base != self._prev_url:
                # An engine Studio started counts from zero, so even an answer that finished before the first
                # sample shows up. A server it attached to has run before: its first sample is the baseline.
                self._prev = {} if state == "ready" else None
                self._prev_url, self.metrics_available = base, None
            code, text = self._get(base, "/metrics", headers)
            if code == 200 and text:
                self.metrics_available = True
                cur = parse_prometheus(text)
                s.update(self._rates(cur))
                self.totals = {"generated": cur.get(P + "tokens_predicted_total", 0.0),
                               "prompt": cur.get(P + "prompt_tokens_total", 0.0),
                               "decode_calls": cur.get(P + "n_decode_total", 0.0)}
            elif code in (404, 501):
                self.metrics_available = False
            if tick % 2 == 0:
                code, text = self._get(base, "/slots", headers)
                if code == 200 and text.startswith("["):
                    try:
                        slots = json.loads(text)
                        s["slots_busy"] = sum(1 for x in slots if x.get("is_processing"))
                        s["slots"] = len(slots)
                    except ValueError:
                        pass
        else:
            self._prev, self._prev_url = None, ""  # the next start (same port or not) begins a new count
        cache = self.engine.cache
        if cache.get("hit_rate") is not None:
            s["cache_hit"] = cache["hit_rate"]
        with self._lock:
            if self.metrics_available is False and self.responses:
                last = self.responses[-1]
                s["last_answer"] = True
                s["decode"] = last.get("predicted_per_second")
                s["prefill"] = last.get("prompt_per_second")
                if last.get("draft_n"):
                    s["accept"] = (last.get("draft_n_accepted") or 0) / last["draft_n"]
            self.history.append(s)

    def _rates(self, cur: dict[str, float]) -> dict[str, Any]:
        prev, self._prev = self._prev, cur
        out: dict[str, Any] = {"busy": cur.get(P + "requests_processing", 0) > 0,
                               "deferred": int(cur.get(P + "requests_deferred", 0))}
        if prev is None:
            return out

        def d(name: str) -> float:
            return cur.get(P + name, 0.0) - prev.get(P + name, 0.0)

        if d("tokens_predicted_total") < 0 or d("prompt_tokens_total") < 0:  # the server restarted
            return out
        gen_n, gen_s = d("tokens_predicted_total"), d("tokens_predicted_seconds_total")
        pp_n, pp_s = d("prompt_tokens_total"), d("prompt_seconds_total")
        drafted, accepted = d("spec_decode_num_draft_tokens_total"), d("spec_decode_num_accepted_tokens_total")
        if gen_n > 0 and gen_s > 0:
            out["decode"] = round(gen_n / gen_s, 2)
            out["generated"] = int(gen_n)
        if pp_n > 0 and pp_s > 0:
            out["prefill"] = round(pp_n / pp_s, 1)
            out["prompt_tokens"] = int(pp_n)
        if drafted > 0:
            out["accept"] = round(accepted / drafted, 4)
        return out

    def snapshot(self, since: float = 0.0) -> dict[str, Any]:
        with self._lock:
            hist = [h for h in self.history if h["t"] > since] if since else list(self.history)
            last_resp = self.responses[-1] if self.responses else None
        return {
            "now": time.time(),
            "history": hist,
            "metrics_available": self.metrics_available,
            "last_response": last_resp,
            "totals": self.totals,
            "cache": dict(self.engine.cache),
            "accept_logged": self.engine.accept_logged,
        }
