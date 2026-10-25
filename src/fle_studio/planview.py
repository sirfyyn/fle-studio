"""Short facts about a plan, for the chip row above the notes in Engine > Advanced.

Pure: it reads the public form of a plan (what `Plan.public()` returns), the model record and two settings, and
returns ``[{"label", "value", "tone"}]`` with tone ``ok`` (as wanted), ``warn`` (something is not as asked, or a
fallback) or ``info`` (a plain fact). Nothing here starts anything or reads the disk."""

from __future__ import annotations

from typing import Any

from . import qol


def _value(args: list[str], *names: str) -> str | None:
    """The value of the first of these flags, in `--flag value` or `--flag=value` form."""
    for i, arg in enumerate(args):
        for name in names:
            if arg == name and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def _int(text: str | None) -> int | None:
    try:
        return int(str(text))
    except (TypeError, ValueError):
        return None


def effective_speculation(plan: dict[str, Any]) -> str:
    """What the command really does: ``mtp``, ``ngram-mod``, ``ngram-simple`` or another --spec-type, else ``none``."""
    args = [str(a) for a in plan.get("args") or []]
    kind = (_value(args, "--spec-type") or "").strip().lower()
    if kind in ("", "none"):
        return "none"
    return "mtp" if kind == "draft-mtp" else kind


def summary(
    plan: dict[str, Any], *, model: dict[str, Any] | None, spec_requested: str = "auto", mtp_on: bool = True
) -> list[dict[str, str]]:
    """The chips for one plan. `spec_requested` and `mtp_on` are the saved Speculation and MTP settings."""
    args = [str(a) for a in plan.get("args") or []]
    chips: list[dict[str, str]] = []

    def add(label: str, value: str, tone: str) -> None:
        chips.append({"label": label, "value": value, "tone": tone})

    fallback = bool(plan.get("fallback"))
    engine = str(plan.get("engine") or "")
    if engine == "fle":
        add("Engine", "FLE", "info")
    elif engine == "upstream":
        add("Engine", "Upstream llama.cpp", "warn" if fallback else "info")
    else:  # a build of your own: what it is shows in the command (the FLE engine's flags and settings)
        env = plan.get("env") or {}
        fle = any(str(k).startswith("FLE_") for k in env) or any(a.startswith("--moe-expert-cache") for a in args)
        add("Engine", "FLE (own build)" if fle else "llama.cpp (own build)", "info")

    spec = effective_speculation(plan)
    asked = str(spec_requested or "auto").lower()
    if spec != "none":
        add("Speculation", spec, "ok")
    elif asked not in ("auto", "none") and plan.get("profile", "auto") == "auto":
        add("Speculation", "none (not as asked)", "warn")
    else:
        add("Speculation", "none", "info")

    mtp = (model or {}).get("mtp") or {}
    if mtp.get("supported"):
        have = bool(mtp.get("head") and mtp.get("vocab"))
        if have:
            add("MTP files", "present", "ok")
        elif mtp_on and asked in ("auto", "mtp") and plan.get("profile", "auto") == "auto":
            add("MTP files", "missing", "warn")
        else:
            add("MTP files", "missing (not needed)", "info")

    context = _int(_value(args, "-c", "--ctx-size"))
    if context and context > 0:
        parallel = max(1, _int(_value(args, "--parallel", "-np")) or 1)
        unified = "--kv-unified" in args or "-kvu" in args
        eff = qol.effective_context(context, parallel, unified)
        add("Context", f"{eff['context']:,} tokens", "info")
        if parallel > 1:
            share = f"up to {eff['per_slot']:,} (shared)" if eff["kv_unified"] else f"{eff['per_slot']:,}"
            add("Per slot", share, "info")
        else:
            add("Per slot", f"{eff['per_slot']:,}", "info")
        add("Slots", str(parallel), "info")

    add("Fallback", "yes" if fallback else "no", "warn" if fallback else "ok")
    return chips
