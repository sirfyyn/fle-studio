"""Helpers shared by the tuner and the A/B/A comparison: change a plan, run one trial, save a plan as a profile.

A *trial* is: start the engine with a plan, wait until it answers, send the speed requests several times, stop the
engine again. Everything that varies between trials is a change to the plan (a flag, an environment variable), never to
the user's settings, so the saved settings stay as they are and a trial can be written out as a plain command.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from collections.abc import Callable
from typing import Any

from . import bench, flaghelp
from .engine import Engine
from .launch import Plan, Profile, _arg_value
from .system import GiB

Flags = dict[str, "str | bool | None"]  # flag -> value; True = a switch without value; None = remove the flag


class TrialFailed(RuntimeError):
    """The engine did not start or did not answer with this plan. The message says why."""


# ------------------------------------------------------------------------------------------------ plan changes


def set_flag(args: list[str], name: str, value: str | bool | None) -> list[str]:
    """``args`` with the flag ``name`` set to ``value`` (any spelling of it is replaced in place, extra copies drop)."""
    names = set(flaghelp.names_of(name))
    info = flaghelp.lookup(name)
    takes = info[1] if info else True
    out: list[str] = []
    done = False
    i = 0
    while i < len(args):
        token = args[i]
        base, eq, _ = token.partition("=") if token.startswith("--") else (token, "", "")
        if base not in names:
            out.append(token)
            i += 1
            continue
        consumed = 1
        if not eq and takes and i + 1 < len(args) and (not args[i + 1].startswith("-") or args[i + 1][1:2].isdigit()):
            consumed = 2
        if not done and value is not None:
            if value is True:
                out.append(base)
            elif eq:
                out.append(f"{base}={value}")
            else:
                out += [base, str(value)]
        done = done or value is not None
        i += consumed
    if not done and value is not None:
        out += [name] if value is True else [name, str(value)]
    return out


def change_plan(plan: Plan, flags: Flags | None = None, env: dict[str, str | None] | None = None) -> Plan:
    """A copy of ``plan`` with flags and environment variables changed (``None`` removes one)."""
    new = dataclasses.replace(plan, args=list(plan.args), env=dict(plan.env), notes=list(plan.notes))
    for name, value in (flags or {}).items():
        new.args = set_flag(new.args, name, value)
    for key, value in (env or {}).items():
        if value is None:
            new.env.pop(key, None)
        else:
            new.env[key] = value
    return new


def flag_value(plan: Plan, name: str) -> str | None:
    """The value of a flag in the plan, whichever way it is spelled."""
    return _arg_value(plan.args, *flaghelp.names_of(name))


def describe(flags: Flags, env: dict[str, str | None] | None = None) -> str:
    parts = [f"{k} {v}" if v is not True else k for k, v in flags.items() if v is not None]
    parts += [f"{k}={v}" for k, v in (env or {}).items() if v is not None]
    return ", ".join(parts)


# ------------------------------------------------------------------------------------------------ profiles


def machine_key(sysinfo: dict[str, Any]) -> str:
    """A short fingerprint of the hardware that settings were measured on: change the GPU or the RAM and a tuned
    profile no longer claims to be the best for this machine."""
    cpu = sysinfo.get("cpu") or {}
    gpus = [
        f"{g.get('vendor')}:{g.get('name')}:{int(g.get('memory_total') or 0) // GiB}" for g in sysinfo.get("gpus") or []
    ]
    parts = [
        str(cpu.get("name") or ""),
        str(cpu.get("physical") or ""),
        str(int((sysinfo.get("memory") or {}).get("total", 0)) // GiB),
        *sorted(gpus),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def profile_from_plan(plan: Plan, *, ident: str, name: str, notes: str, meta: dict[str, Any]) -> Profile:
    """A saved profile that starts exactly like ``plan``: paths of the model, MTP files and expert profile become
    placeholders again, so the profile keeps working when the model is somewhere else or another one is picked.

    The arguments FLE Studio adds on its own (host, port, metrics, alias, log level) are left out; it adds them
    again."""
    model = plan.model or {}
    mtp = model.get("mtp") or {}
    swap = {
        str(model.get("path") or ""): "{model}",
        str(mtp.get("head") or ""): "{mtp_head}",
        str(mtp.get("vocab") or ""): "{mtp_vocab}",
    }
    swap.pop("", None)

    def back(text: str) -> str:
        if text in swap:
            return swap[text]
        return text

    args = list(plan.args)
    for flag in ("--host", "--port", "-a", "-lv"):
        args = set_flag(args, flag, None)
    args = set_flag(args, "--metrics", None)
    args = [back(a) for a in args]
    env = {k: ("{expert_profile}" if k == "FLE_MOE_PROFILE" else back(v)) for k, v in plan.env.items()}
    custom = plan.engine == "custom"
    return Profile(
        id=ident,
        name=name,
        engine="custom" if custom else plan.engine,
        binary=plan.binary if custom else "",
        args=args,
        env=env,
        notes=notes,
        meta=meta,
    )


# ------------------------------------------------------------------------------------------------ one trial


class TrialRunner:
    """Starts the engine with a plan, measures it, stops it. One trial at a time, blocking."""

    def __init__(
        self, engine: Engine, runner: bench.Bench, cancel: threading.Event, ready_timeout: float = 900.0
    ) -> None:
        self.engine = engine
        self.bench = runner
        self.cancel = cancel
        self.ready_timeout = ready_timeout

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise InterruptedError

    def run(
        self,
        plan: Plan,
        runs: int,
        tests: tuple[dict[str, Any], ...] = bench.QUICK_TESTS,
        on_first_run: Callable[[list[dict[str, Any]]], None] | None = None,
    ) -> dict[str, Any]:
        """Returns {load_s, seconds, tests, decode_runs, prefill_runs, decode_ci, prefill_ci, accept?}.

        Raises TrialFailed when the engine does not come up or answers nothing (the engine is stopped in every case),
        InterruptedError when cancelled. ``on_first_run`` sees the results of the first run before the others start;
        the tuner's probe uses it to give up early."""
        t0 = time.monotonic()
        self.engine.start(plan)
        try:
            self._wait_ready()
            load_s = round(time.monotonic() - t0, 1)
            state, base, headers = self.engine.target()[:3]
            results: list[dict[str, Any]] = []
            for run in range(runs):
                self._check_cancel()
                part = self.bench.measure(base, headers, 1, tests, cancel=self.cancel, run_offset=run)
                results += part
                if run == 0 and on_first_run:
                    on_first_run(part)
            return {
                "load_s": load_s,
                "seconds": round(time.monotonic() - t0, 1),
                "tests": results,
                **bench.run_stats(results),
            }
        except InterruptedError:  # a subclass of OSError, so it must be let through before the next clause
            raise
        except (OSError, ValueError) as exc:  # urllib errors, a stream that ended early
            raise TrialFailed(f"The engine stopped answering: {getattr(exc, 'reason', exc)}") from None
        finally:
            self.stop()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.ready_timeout
        while True:
            self._check_cancel()
            state = self.engine.state
            if state in ("ready", "attached"):
                return
            if state in ("crashed", "stopped"):
                error = self.engine.error or "The engine stopped while starting"
                hint = f" {self.engine.hint}" if self.engine.hint else ""
                raise TrialFailed(f"{error}{hint}")
            if time.monotonic() > deadline:
                raise TrialFailed(f"The engine did not become ready within {self.ready_timeout:.0f} s")
            time.sleep(0.2)

    def stop(self) -> None:
        """Stop the engine and wait until the process is gone, so the next trial gets all of its memory."""
        self.engine.stop()
        deadline = time.monotonic() + 60
        while self.engine.state in ("starting", "ready", "stopping") and time.monotonic() < deadline:
            time.sleep(0.1)
