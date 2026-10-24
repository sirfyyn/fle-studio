"""A look into the FLE expert cache: which experts are hot, how the cache fills, and the hit rate per layer.

The data comes from the lines the FLE engine prints to its log (FLE Studio follows that log for the loading phases
and the overall hit rate already; for a systemd service it is the journal). Three line types are understood, the
format is written down in docs/EXPERT-CACHE-DATA.md:

    MoE expert cache enabled: 40 layers x 96 slots, 1 inserts/step, 14984.0 MiB device memory      (exists today)
    moe-cache: steps=512 hits=123 misses=45 hit-rate=73.2%                                         (exists today)
    moe-layer: steps=512 layer=7 hits=1234 misses=210 filled=88 slots=96                           (new)
    moe-hot: steps=512 layer=7 top=311:812,27:790,402:655                                          (new)

The first two give the overall hit rate over time and the size of the cache. The last two are what the per-layer view
and the hot-expert map need; the engine does not print them yet, the stand-in engine in the tests does. A view that
has no data says what is missing instead of showing an empty chart.

All counts in the lines are cumulative since the start of the engine, so the view subtracts neighbours to get the hit
rate of an interval. A new "cache enabled" line, or steps that go backwards, start a new run.
"""

from __future__ import annotations

import collections
import re
import threading
import time
from typing import Any

ENABLED_RE = re.compile(r"MoE expert cache enabled[:,]?\s*(?P<text>.*)", re.I)
SIZE_RE = re.compile(r"(\d+)\s+layers\s*x\s*(\d+)\s+slots", re.I)
OVERALL_RE = re.compile(r"moe-cache:.*?(?:steps=(\d+).*?)?hits=(\d+).*?misses=(\d+).*?hit-rate=([\d.]+)\s*%")
LAYER_RE = re.compile(
    r"moe-layer:\s+steps=(\d+)\s+layer=(\d+)\s+hits=(\d+)\s+misses=(\d+)(?:\s+filled=(\d+))?(?:\s+slots=(\d+))?"
)
HOT_RE = re.compile(r"moe-hot:\s+steps=(\d+)\s+layer=(\d+)\s+top=([\d:,]*)")

MAX_POINTS = 400  # per series; older points drop out


def _rate(hits: float, misses: float) -> float | None:
    total = hits + misses
    return round(100.0 * hits / total, 2) if total > 0 else None


class CacheWatch:
    """Collects the cache lines of the running engine. ``feed`` takes every log line, ``snapshot`` is the API answer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.layers_total = 0
            self.slots_per_layer = 0
            self.enabled_text = ""
            self.steps = 0
            self.overall: collections.deque[dict[str, Any]] = collections.deque(maxlen=MAX_POINTS)
            self.layer_points: dict[int, collections.deque[dict[str, Any]]] = {}
            self.hot: dict[int, list[tuple[int, int]]] = {}
            self.hot_steps = 0
            self.lines_seen = 0

    # ------------------------------------------------------------------------------------------ input

    def feed(self, line: str, now: float | None = None) -> bool:
        """Read one engine log line; True when it was a cache line (or the start of a new run)."""
        if line.startswith("$ "):  # the command line FLE Studio logs when it starts an engine: a new run
            self.reset()
            return True
        if "moe-" not in line and "MoE expert cache" not in line:
            return False
        now = time.time() if now is None else now
        m = ENABLED_RE.search(line)
        if m:
            self.reset()
            with self._lock:
                self.enabled_text = m.group("text").strip()
                size = SIZE_RE.search(self.enabled_text)
                if size:
                    self.layers_total, self.slots_per_layer = int(size.group(1)), int(size.group(2))
                self.lines_seen += 1
            return True
        m = OVERALL_RE.search(line)
        if m:
            self._overall(int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), float(m.group(4)), now)
            return True
        m = LAYER_RE.search(line)
        if m:
            filled = int(m.group(5)) if m.group(5) is not None else None
            slots = int(m.group(6)) if m.group(6) is not None else None
            self._layer(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), filled, slots, now)
            return True
        m = HOT_RE.search(line)
        if m:
            top = []
            for part in m.group(3).split(","):
                expert, _, count = part.partition(":")
                if expert.isdigit() and count.isdigit():
                    top.append((int(expert), int(count)))
            self._hot(int(m.group(1)), int(m.group(2)), top)
            return True
        return False

    def _restart_if_steps_went_back(self, steps: int) -> None:
        if steps and steps < self.steps:
            with self._lock:
                enabled, layers, slots = self.enabled_text, self.layers_total, self.slots_per_layer
            self.reset()
            with self._lock:
                self.enabled_text, self.layers_total, self.slots_per_layer = enabled, layers, slots

    def _overall(self, steps: int, hits: int, misses: int, rate: float, now: float) -> None:
        self._restart_if_steps_went_back(steps)
        with self._lock:
            prev = self.overall[-1] if self.overall else None
            point = {
                "t": round(now, 2),
                "steps": steps,
                "hits": hits,
                "misses": misses,
                "hit_rate": rate,
                "recent_rate": _rate(hits - prev["hits"], misses - prev["misses"]) if prev else rate,
            }
            self.overall.append(point)
            self.steps = max(self.steps, steps)
            self.lines_seen += 1

    def _layer(
        self, steps: int, layer: int, hits: int, misses: int, filled: int | None, slots: int | None, now: float
    ) -> None:
        self._restart_if_steps_went_back(steps)
        with self._lock:
            series = self.layer_points.setdefault(layer, collections.deque(maxlen=MAX_POINTS))
            prev = series[-1] if series else None
            series.append(
                {
                    "t": round(now, 2),
                    "steps": steps,
                    "hits": hits,
                    "misses": misses,
                    "filled": filled,
                    "slots": slots,
                    "hit_rate": _rate(hits, misses),
                    "recent_rate": _rate(hits - prev["hits"], misses - prev["misses"]) if prev else None,
                }
            )
            self.steps = max(self.steps, steps)
            self.lines_seen += 1

    def _hot(self, steps: int, layer: int, top: list[tuple[int, int]]) -> None:
        with self._lock:
            self.hot[layer] = top
            self.hot_steps = max(self.hot_steps, steps)
            self.lines_seen += 1

    # ------------------------------------------------------------------------------------------ output

    def snapshot(self, layer: int | None = None) -> dict[str, Any]:
        """Everything the Cache page shows. ``layer`` adds the time series of that one layer."""
        with self._lock:
            overall = list(self.overall)
            latest = {n: s[-1] for n, s in sorted(self.layer_points.items()) if s}
            layer_series = list(self.layer_points.get(layer, [])) if layer is not None else []
            hot = {n: list(top) for n, top in sorted(self.hot.items())}
            fill = self._fill_series_locked()
            out: dict[str, Any] = {
                "available": {"overall": bool(overall), "layers": bool(latest), "hot": bool(hot)},
                "enabled": {"layers": self.layers_total, "slots": self.slots_per_layer, "text": self.enabled_text}
                if self.enabled_text
                else None,
                "steps": self.steps,
                "lines": self.lines_seen,
            }
        out["overall"] = {
            "series": overall,
            "hit_rate": overall[-1]["hit_rate"] if overall else None,
            "recent_rate": overall[-1]["recent_rate"] if overall else None,
        }
        layers = []
        for n, p in latest.items():
            slots = p["slots"] or self.slots_per_layer or None
            layers.append(
                {
                    "layer": n,
                    "hits": p["hits"],
                    "misses": p["misses"],
                    "hit_rate": p["hit_rate"],
                    "recent_rate": p["recent_rate"],
                    "filled": p["filled"],
                    "slots": slots,
                    "fill": round(p["filled"] / slots, 4) if p["filled"] is not None and slots else None,
                }
            )
        out["layers"] = layers
        out["fill"] = {"series": fill, "now": fill[-1]["fill"] if fill else None}
        out["hot"] = {"steps": self.hot_steps, "layers": {str(n): top for n, top in hot.items()}}
        if layer is not None:
            out["layer_series"] = {"layer": layer, "series": layer_series}
        out["missing"] = self.missing(out["available"])
        return out

    def _fill_series_locked(self) -> list[dict[str, Any]]:
        """The share of cache slots in use over time: over all layers that reported at the same step count."""
        by_steps: dict[int, list[tuple[int, int, float]]] = {}
        for series in self.layer_points.values():
            for p in series:
                if p["filled"] is not None and (p["slots"] or self.slots_per_layer):
                    by_steps.setdefault(p["steps"], []).append(
                        (p["filled"], p["slots"] or self.slots_per_layer, p["t"])
                    )
        out = []
        for steps in sorted(by_steps):
            rows = by_steps[steps]
            total_slots = sum(r[1] for r in rows)
            out.append(
                {
                    "steps": steps,
                    "t": max(r[2] for r in rows),
                    "fill": round(sum(r[0] for r in rows) / total_slots, 4),
                    "layers": len(rows),
                }
            )
        return out[-MAX_POINTS:]

    @staticmethod
    def missing(available: dict[str, bool]) -> list[str]:
        """What the engine does not print (yet), in words for the page."""
        out = []
        if not available["overall"]:
            out.append(
                "The overall hit rate (the `moe-cache:` line) has not appeared yet. The engine prints it every "
                "512 decode steps, and only at log verbosity 4 (FLE Studio starts it with `-lv 4`)."
            )
        if not available["layers"]:
            out.append(
                "Per-layer hit rate and fill need `moe-layer:` lines, which this engine does not print yet "
                "(format: docs/EXPERT-CACHE-DATA.md)."
            )
        if not available["hot"]:
            out.append(
                "The hot experts need `moe-hot:` lines, which this engine does not print yet "
                "(format: docs/EXPERT-CACHE-DATA.md)."
            )
        return out
