"""Set the engine up by measuring instead of guessing: the auto-tuner and the A/B/A comparison.

Both start the engine themselves, one setting after the other, send the speed requests several times and stop it again
(see trials.py), so the engine must be idle when they begin and is stopped when they end.

**The tuner** changes one thing at a time, starting from the settings FLE Studio would use anyway:

1. *Probe.* The starting settings run once. If the engine does not come up or answers nothing, the tuner stops here
   and says why; nothing else is tried.
2. For each setting that applies (threads, the GPU expert cache of the FLE engine or ``--n-cpu-moe`` of llama.cpp,
   batch size, optionally context size) a few values around the current one are tried, each as a full trial with
   several runs. A value replaces the current best only if it is faster with a 95 % interval that excludes 0
   (stats.compare); a difference inside the noise keeps what is there. A value that does not even load (out of memory)
   is recorded and skipped.
3. A time budget can be set. Before every trial the tuner checks that the longest trial so far still fits into what is
   left; it never starts one it cannot finish.
4. The winner is saved as a launch profile (``Tuned: <model>``) that starts exactly like the winning plan; "Best for my
   machine" on the Engine page uses it. The profile remembers the machine it was measured on.

For llama.cpp (not the FLE engine) the tuner first screens thread count, batch and ``--n-cpu-moe`` with one
``llama-bench`` call each, if that tool is there: it needs no engine restarts and picks the most promising value, which
then still has to win a normal trial against the current best.

**A/B/A** measures setting A, then B, then A again with the same number of runs each, and tells whether B is faster,
slower or within noise (stats.aba). Measuring A twice shows drift, such as a warming CPU, that a plain A-then-B run
would call a difference.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import bench, engine_tools, paths, stats
from .engine import Engine, EngineBusy
from .launch import Plan, Profiles
from .trials import Flags, TrialFailed, TrialRunner, change_plan, describe, flag_value, machine_key, profile_from_plan

CACHE_ENV = "FLE_MOE_CACHE_BUDGET_MIB"
PARAM_LABELS = {
    "threads": "CPU threads",
    "cache": "GPU expert cache",
    "ncmoe": "Experts kept on the CPU (--n-cpu-moe)",
    "batch": "Batch size",
    "context": "Context size",
}
DEFAULT_PARAMS = ("threads", "cache", "ncmoe", "batch")
MIN_RUNS, MAX_RUNS = 2, 10


# ------------------------------------------------------------------------------------------------ what to try


@dataclass
class Change:
    label: str
    flags: Flags = field(default_factory=dict)
    env: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Param:
    key: str
    objective: str  # decode | prefill: which speed decides
    rule: str  # speed: must be faster; capacity: bigger is better as long as it is not slower
    base: str  # the value of the starting settings, in words
    values: list[Change]

    @property
    def label(self) -> str:
        return PARAM_LABELS[self.key]


def _round(n: float, step: int = 256) -> int:
    return max(step, int(n // step) * step)


def candidates(plan: Plan, sysinfo: dict[str, Any], keys: tuple[str, ...] = DEFAULT_PARAMS) -> list[Param]:
    """The settings worth trying for this plan on this machine, in the order they are tried. Settings that do not
    apply (the FLE cache without the FLE engine, ``--n-cpu-moe`` with it, a GPU setting without a GPU) are left out."""
    info = (plan.model or {}).get("info") or {}
    physical = int((sysinfo.get("cpu") or {}).get("physical") or 1)
    fle = CACHE_ENV in plan.env
    has_gpu = any(g.get("vendor") != "none" for g in sysinfo.get("gpus") or [])
    out: list[Param] = []
    for key in keys:
        if key == "threads" and physical > 1:
            base = flag_value(plan, "-t")
            values = sorted({max(1, physical - 1), physical, max(1, physical // 2)} - {int(base) if base else -1})
            out.append(
                Param(
                    key,
                    "decode",
                    "speed",
                    base or "default",
                    [Change(f"{v} threads", {"-t": str(v)}) for v in values if v != int(base or 0)],
                )
            )
        elif key == "cache" and fle:
            budget = int(plan.env[CACHE_ENV] or 0)
            if budget >= 1024:
                vals = [_round(budget * 0.8), _round(budget * 1.1)]
                out.append(
                    Param(
                        key,
                        "decode",
                        "speed",
                        f"{budget} MiB",
                        [Change(f"expert cache {v} MiB", env={CACHE_ENV: str(v)}) for v in vals if v != budget],
                    )
                )
        elif key == "ncmoe" and not fle and has_gpu and info.get("moe") and info.get("layers"):
            if flag_value(plan, "--n-cpu-moe") is None and "--cpu-moe" not in plan.args:
                layers = int(info["layers"])
                vals = sorted({layers // 4, layers // 2, (3 * layers) // 4} - {0})
                out.append(
                    Param(
                        key,
                        "decode",
                        "speed",
                        "chosen by -fit",
                        [Change(f"{v} layers of experts on the CPU", {"--n-cpu-moe": str(v)}) for v in vals],
                    )
                )
        elif key == "batch":
            base = flag_value(plan, "-b")
            vals = [v for v in (512, 1024, 2048, 4096) if str(v) != base]
            if "-ub" in plan.args or "--ubatch-size" in plan.args:
                ub = flag_value(plan, "-ub")
                vals = [v for v in vals if str(v) != ub]
            out.append(
                Param(
                    key,
                    "prefill",
                    "speed",
                    base or "default",
                    [Change(f"batch {v}", {"-b": str(v), "-ub": str(v)}) for v in vals],
                )
            )
        elif key == "context":
            ctx = int(flag_value(plan, "-c") or 0)
            limit = int(info.get("context_length") or 0)
            vals = [v for v in (ctx * 2, ctx * 4) if ctx and (not limit or v <= limit)]
            out.append(
                Param(
                    key,
                    "decode",
                    "capacity",
                    f"{ctx:,} tokens",
                    [Change(f"context {v:,} tokens", {"-c": str(v)}) for v in vals],
                )
            )
    return [p for p in out if p.values]


def parse_bench_json(text: str) -> list[dict[str, Any]]:
    """The entries of ``llama-bench -o json``: [{n_threads, n_batch, n_prompt, n_gen, samples_ts, avg_ts, ...}]."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("llama-bench printed no JSON")
    rows = json.loads(text[start : end + 1])
    if not isinstance(rows, list):
        raise ValueError("llama-bench printed no list")
    return rows


def screen_values(rows: list[dict[str, Any]], field_name: str) -> dict[int, dict[str, Any]]:
    """Per value of one parameter (``n_threads``, ``n_batch``, ``n_cpu_moe``): the generation and prompt speeds."""
    out: dict[int, dict[str, Any]] = {}
    for r in rows:
        if field_name not in r:
            continue
        slot = out.setdefault(int(r[field_name]), {})
        key = "prefill" if r.get("n_prompt") else "decode"
        slot[key] = list(r.get("samples_ts") or [r.get("avg_ts")])
    return out


# ------------------------------------------------------------------------------------------------ the host


@dataclass
class Host:
    """What the tuner needs from the rest of FLE Studio, so it can be tested without a server around it."""

    engine: Engine
    bench: bench.Bench
    profiles: Profiles
    data_dir: Path
    make_plan: Callable[[str, str | None], Plan]
    sysinfo: Callable[[], dict[str, Any]]
    ready_timeout: float = 900.0
    on_saved: Callable[[], None] | None = None


class Tuner:
    def __init__(self, host: Host) -> None:
        self.host = host
        self.dir = paths.ensure(host.data_dir / "tuning")
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self.current: dict[str, Any] | None = None

    # ---- results

    def results(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.dir.glob("*.json"), reverse=True)[:100]:
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out

    def delete(self, ident: str) -> None:
        for p in self.dir.glob(f"*-{ident}.json"):
            p.unlink()

    def status(self) -> dict[str, Any] | None:
        with self._lock:
            return json.loads(json.dumps(self.current)) if self.current else None

    def cancel(self) -> None:
        self._cancel.set()

    def tuned_profile(self, model_id: str) -> dict[str, Any] | None:
        """The profile the tuner saved for this model on this machine, or None (it was measured elsewhere, or never)."""
        key = machine_key(self.host.sysinfo())
        for p in self.host.profiles.all():
            meta = (p.meta or {}).get("tuned") or {}
            if meta.get("model_id") == model_id and meta.get("machine") == key:
                return p.public()
        return None

    # ---- starting

    def _begin(self, kind: str, fields: dict[str, Any]) -> dict[str, Any]:
        engine = self.host.engine
        with self._lock:
            if self.current and self.current["state"] == "running":
                raise RuntimeError("A tuning run or comparison is already running")
            if engine.state not in ("stopped", "crashed") or engine.unit is not None:
                raise EngineBusy(
                    "Stop the engine first (or disconnect from the server): the tuner starts the engine "
                    "itself, once for every setting it tries, and stops it at the end."
                )
            self._cancel.clear()
            self.current = {
                "id": secrets.token_hex(4),
                "kind": kind,
                "state": "running",
                "started": time.time(),
                "step": "Starting",
                "trials": [],
                "notes": [],
                **fields,
            }
            return self.current

    def start_tune(
        self,
        *,
        model_id: str,
        profile: str | None = None,
        budget_s: int = 1200,
        runs: int = 3,
        params: tuple[str, ...] = DEFAULT_PARAMS,
        screen: bool = True,
    ) -> dict[str, Any]:
        runs = max(MIN_RUNS, min(MAX_RUNS, int(runs)))
        budget_s = max(60, min(86400, int(budget_s)))
        keys = tuple(p for p in params if p in PARAM_LABELS) or DEFAULT_PARAMS
        base_plan = self.host.make_plan(model_id, profile)  # raises PlanError/ApiError with a readable message
        cur = self._begin(
            "tune",
            {
                "model": {"id": model_id, "name": base_plan.model.get("name", "")},
                "profile": profile or "auto",
                "budget_s": budget_s,
                "runs": runs,
                "params": list(keys),
                "screen": bool(screen),
            },
        )
        threading.Thread(target=self._run_tune, args=(cur, base_plan, keys, bool(screen)), daemon=True).start()
        return dict(cur)

    def start_aba(
        self, *, model_id: str, profile_a: str, profile_b: str, runs: int = 3, label: str = ""
    ) -> dict[str, Any]:
        runs = max(MIN_RUNS, min(MAX_RUNS, int(runs)))
        plan_a, plan_b = self.host.make_plan(model_id, profile_a), self.host.make_plan(model_id, profile_b)
        cur = self._begin(
            "aba",
            {
                "model": {"id": model_id, "name": plan_a.model.get("name", "")},
                "a": {"profile": profile_a, "command": plan_a.command_line()},
                "b": {"profile": profile_b, "command": plan_b.command_line()},
                "runs": runs,
                "label": label.strip()[:80],
            },
        )
        threading.Thread(target=self._run_aba, args=(cur, plan_a, plan_b), daemon=True).start()
        return dict(cur)

    # ---- shared pieces

    def _set(self, cur: dict[str, Any], **fields: Any) -> None:
        with self._lock:
            cur.update(fields)

    def _finish(self, cur: dict[str, Any], state: str, error: str = "") -> None:
        with self._lock:
            cur["state"], cur["finished"], cur["step"] = state, time.time(), ""
            cur["elapsed"] = round(cur["finished"] - cur["started"], 1)
            if error:
                cur["error"] = error
            record = json.loads(json.dumps(cur))
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(record["started"])) + f"-{record['id']}.json"
        (self.dir / name).write_text(json.dumps(record, indent=1), encoding="utf-8")

    def _trial(
        self,
        cur: dict[str, Any],
        plan: Plan,
        runs: int,
        what: str,
        on_first_run: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        self._set(cur, step=what)
        runner = TrialRunner(self.host.engine, self.host.bench, self._cancel, self.host.ready_timeout)
        return runner.run(plan, runs, bench.QUICK_TESTS, on_first_run)

    @staticmethod
    def _slim(res: dict[str, Any]) -> dict[str, Any]:
        """A trial without the raw request list: what the page and the stored record keep."""
        return {
            "load_s": res["load_s"],
            "seconds": res["seconds"],
            "decode_runs": res["decode_runs"],
            "prefill_runs": res["prefill_runs"],
            "decode": res["decode_ci"],
            "prefill": res["prefill_ci"],
        }

    # ---- the tuner

    def _run_tune(self, cur: dict[str, Any], base_plan: Plan, keys: tuple[str, ...], screen: bool) -> None:
        engine = self.host.engine
        restart = engine.restart_on_crash
        engine.restart_on_crash = False  # a candidate that crashes is a result, not a reason to start it again
        try:
            sysinfo = self.host.sysinfo()
            runs = cur["runs"]
            t0 = time.monotonic()

            def left() -> float:
                return cur["budget_s"] - (time.monotonic() - t0)

            # 1. the probe: the starting settings, first run checked at once
            def check(part: list[dict[str, Any]]) -> None:
                if not part or not any((r.get("decode") or 0) > 0 for r in part):
                    raise TrialFailed(
                        "The engine answered, but generated no tokens. Check the model and the engine "
                        "log on the Engine page."
                    )

            try:
                first = self._trial(cur, base_plan, runs, "Probe: the starting settings", check)
            except TrialFailed as exc:
                self._set(cur, probe={"ok": False, "error": str(exc)})
                self._finish(cur, "failed", f"The starting settings do not work, so nothing was tried: {exc}")
                return
            best_plan, best = base_plan, first
            longest = first["seconds"]
            self._set(
                cur,
                probe={"ok": True, **self._slim(first)},
                baseline=self._slim(first),
                best={"label": "starting settings", "command": base_plan.command_line(), **self._slim(first)},
                trial_seconds=longest,
            )
            bench_tool = (
                engine_tools.find_tool(base_plan.binary, "llama-bench") if screen and CACHE_ENV not in base_plan.env else None
            )
            params = candidates(base_plan, sysinfo, keys)
            cur_notes: list[str] = []
            if not params:
                cur_notes.append("Nothing to try for this model and engine; the starting settings stay.")
            stopped = ""
            for param in params:
                values = list(param.values)
                if bench_tool is not None and param.key in ("threads", "batch", "ncmoe") and len(values) > 1:
                    values = self._screen(cur, bench_tool, best_plan, param, values) or values
                for change in values:
                    if left() < longest * 1.1:
                        stopped = "The time budget ran out"
                        break
                    plan = change_plan(best_plan, change.flags, change.env)
                    entry: dict[str, Any] = {
                        "n": len(cur["trials"]) + 1,
                        "param": param.key,
                        "param_label": param.label,
                        "label": change.label,
                        "change": describe(change.flags, change.env),
                    }
                    try:
                        res = self._trial(cur, plan, runs, f"{param.label}: {change.label}")
                    except TrialFailed as exc:
                        entry.update(state="failed", error=str(exc), kept=False)
                        with self._lock:
                            cur["trials"].append(entry)
                        continue
                    longest = max(longest, res["seconds"])
                    key = "decode_runs" if param.objective == "decode" else "prefill_runs"
                    cmp = stats.compare(best[key], res[key])
                    take = cmp["verdict"] == "better" or (param.rule == "capacity" and cmp["verdict"] == "within noise")
                    entry.update(
                        state="ok",
                        kept=take,
                        vs_best={k: cmp[k] for k in ("verdict", "pct", "pct_lo", "pct_hi", "method")},
                        **self._slim(res),
                    )
                    with self._lock:
                        cur["trials"].append(entry)
                    if take:
                        best_plan, best = plan, res
                        self._set(cur, best={"label": change.label, "command": plan.command_line(), **self._slim(res)})
                if stopped:
                    break
            if stopped:
                cur_notes.append(f"{stopped}; the values not tried yet stay as they are.")
            self._set(cur, notes=cur_notes)
            profile = self._save_profile(cur, best_plan, base_plan, first, best)
            self._set(cur, profile=profile, best={**cur["best"], "command": best_plan.command_line()})
            self._finish(cur, "done")
        except InterruptedError:
            self._finish(cur, "cancelled")
        except EngineBusy as exc:
            self._finish(cur, "failed", str(exc))
        except Exception as exc:  # report, never crash the thread silently
            self._finish(cur, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            engine.restart_on_crash = restart
            TrialRunner(engine, self.host.bench, self._cancel).stop()

    def _screen(
        self, cur: dict[str, Any], binary: Path, plan: Plan, param: Param, values: list[Change]
    ) -> list[Change] | None:
        """One llama-bench call over all values of a parameter; returns the most promising one as the only value to
        try, or None (keep all) when the tool failed."""
        field_name, flag = {
            "threads": ("n_threads", "-t"),
            "batch": ("n_batch", "-b"),
            "ncmoe": ("n_cpu_moe", "-ncmoe"),
        }[param.key]
        numbers = []
        for v in values:
            raw = v.flags.get("-t") or v.flags.get("-b") or v.flags.get("--n-cpu-moe")
            numbers.append(str(raw))
        args = [
            "-m",
            plan.model.get("path", ""),
            flag,
            ",".join(numbers),
            "-p",
            "256",
            "-n",
            "32",
            "-r",
            "3",
            "-o",
            "json",
        ]
        self._set(cur, step=f"Screening {param.label} with llama-bench")
        try:
            rc, output = engine_tools.run_tool(binary, args, cancel=self._cancel, timeout=600)
            rows = parse_bench_json(output)
        except (ValueError, OSError, TimeoutError) as exc:
            with self._lock:
                cur["notes"].append(
                    f"llama-bench could not screen {param.label.lower()} ({exc}); every value is tried instead."
                )
            return None
        if rc != 0:
            return None
        table = screen_values(rows, field_name)
        key = "decode" if param.objective == "decode" else "prefill"
        scored = []
        for v, number in zip(values, numbers, strict=True):
            samples = (table.get(int(number)) or {}).get(key) or []
            if samples:
                scored.append((stats.median(samples), v))
        if not scored:
            return None
        scored.sort(key=lambda s: -s[0])
        with self._lock:
            cur.setdefault("screened", []).append(
                {
                    "param": param.key,
                    "values": [{"label": v.label, key: round(m, 2)} for m, v in scored],
                    "picked": scored[0][1].label,
                }
            )
        return [scored[0][1]]

    def _save_profile(
        self, cur: dict[str, Any], best_plan: Plan, base_plan: Plan, first: dict[str, Any], best: dict[str, Any]
    ) -> dict[str, Any]:
        model = base_plan.model
        sysinfo = self.host.sysinfo()
        decode = best["decode_ci"]
        gain = ""
        if best is not first and first["decode_ci"]["median"]:
            gain = (
                f", {100 * (decode['median'] / first['decode_ci']['median'] - 1):+.1f} % against the starting settings"
            )
        notes = (
            f"Tuned by FLE Studio on {time.strftime('%Y-%m-%d')}: {decode['median']:g} tok/s decode "
            f"(median of {decode['n']} runs{gain}). {len(cur['trials'])} settings tried in "
            f"{(time.time() - cur['started']) / 60:.0f} min. Measured on this machine only."
        )
        meta = {
            "tuned": {
                "model_id": model.get("id", cur["model"]["id"]),
                "model": model.get("name", ""),
                "machine": machine_key(sysinfo),
                "date": time.strftime("%Y-%m-%d"),
                "decode": decode,
                "base_decode": first["decode_ci"],
                "trials": len(cur["trials"]),
            }
        }
        ident = f"tuned-{cur['model']['id']}"
        draft = profile_from_plan(
            best_plan, ident=ident, name=f"Tuned: {model.get('name') or cur['model']['id']}", notes=notes, meta=meta
        )
        saved = self.host.profiles.save(
            {
                "id": draft.id,
                "name": draft.name,
                "engine": draft.engine,
                "binary": draft.binary,
                "args": draft.args,
                "env": draft.env,
                "notes": draft.notes,
                "meta": draft.meta,
            }
        )
        if self.host.on_saved:
            self.host.on_saved()
        return {"id": saved.id, "name": saved.name}

    # ---- A/B/A

    def _run_aba(self, cur: dict[str, Any], plan_a: Plan, plan_b: Plan) -> None:
        engine = self.host.engine
        restart = engine.restart_on_crash
        engine.restart_on_crash = False
        try:
            runs = cur["runs"]
            phases = []
            for name, plan in (("A1", plan_a), ("B", plan_b), ("A2", plan_a)):
                res = self._trial(cur, plan, runs, f"{name}: {cur['b' if name == 'B' else 'a']['profile']}")
                phases.append({"phase": name, **self._slim(res)})
                with self._lock:
                    cur["trials"] = list(phases)
            by = {p["phase"]: p for p in phases}
            result = {
                "decode": stats.aba(by["A1"]["decode_runs"], by["B"]["decode_runs"], by["A2"]["decode_runs"]),
                "prefill": stats.aba(by["A1"]["prefill_runs"], by["B"]["prefill_runs"], by["A2"]["prefill_runs"]),
            }
            self._set(cur, result=result)
            self._finish(cur, "done")
        except InterruptedError:
            self._finish(cur, "cancelled")
        except TrialFailed as exc:
            self._finish(cur, "failed", f"{cur.get('step') or 'A trial'} failed: {exc}")
        except Exception as exc:
            self._finish(cur, "failed", f"{type(exc).__name__}: {exc}")
        finally:
            engine.restart_on_crash = restart
            TrialRunner(engine, self.host.bench, self._cancel).stop()
