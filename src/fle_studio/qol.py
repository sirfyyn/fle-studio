"""Setup checks, model verification and the redacted diagnostic report.

The engine helper (`fle`) has the same checks. FLE Studio shows the results here so a user does not need a terminal to
find out why a start would fail.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from . import __version__, gguf, launch

# Type ids used by the engine builds. The last four differ between the FLE release and the DeepSeek41 port, so their
# byte size is not trusted here; the header offsets are enough for the histogram.
TYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
    42: "Q2_0",
    43: "MXFP8/IQ2_KT",
    44: "IQ2_KT/IQ3_KT",
    45: "IQ3_KT/IQ4_KT",
    46: "IQ4_KT",
}

STATUS_ORDER = {"ok": 0, "warn": 1, "fail": 2}
AMBIGUOUS_TYPES = {43, 44, 45, 46}


def _check(name: str, status: str, detail: str, hint: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def _worst(checks: list[dict[str, str]]) -> str:
    out = "ok"
    for item in checks:
        if STATUS_ORDER.get(item["status"], 0) > STATUS_ORDER[out]:
            out = item["status"]
    return out


def _type_name(type_id: int) -> str:
    return TYPE_NAMES.get(type_id, f"type-{type_id}")


def _model_parts(model: dict[str, Any]) -> list[Path]:
    return gguf.split_parts(Path(model.get("path") or ""))


def verify(model: dict[str, Any] | None) -> dict[str, Any]:
    """Read every part header and check the split, tensor offsets, counts and types."""
    if not model:
        return {"ok": False, "errors": ["No model selected."], "warnings": [], "checks": []}
    parts = _model_parts(model)
    present = [p for p in parts if p.exists()]
    missing = [p for p in parts if not p.exists()]
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, str]] = []
    histogram: dict[str, dict[str, int]] = {}
    unknown: set[int] = set()
    ambiguous: set[int] = set()
    duplicate: set[str] = set()
    names: set[str] = set()
    split_counts: set[int] = set()
    archs: set[str] = set()
    incomplete_parts = 0
    total = 0

    if not parts:
        return {"ok": False, "errors": ["The model path is empty."], "warnings": [], "checks": []}
    if not present or present[0] != parts[0]:
        errors.append("The first part of the model is missing.")
        checks.append(_check("first part", "fail", "missing", "Wait for the download to finish."))
    if missing:
        errors.append(f"{len(missing)} of {len(parts)} parts are missing.")
        checks.append(_check("all parts", "fail", f"{len(missing)} missing", "Wait for the download to finish."))
    else:
        checks.append(_check("all parts", "ok", f"{len(parts)} part(s) present"))

    header_by_part: list[gguf.Header] = []
    for part in present:
        index = parts.index(part)
        try:
            header = gguf.read_header(part)
            header_by_part.append(header)
            total += len(header.tensors)
            split_no = header.meta.get("split.no")
            split_count = header.meta.get("split.count")
            split_tensors = header.meta.get("split.tensors.count")
            if isinstance(split_no, int) and split_no != index:
                warnings.append(f"{part.name}: split.no is {split_no}, expected {index}")
            if isinstance(split_count, int):
                split_counts.add(split_count)
            if isinstance(header.meta.get("general.architecture"), str):
                archs.add(str(header.meta["general.architecture"]))
            payload = max(0, header.file_size - header.data_start)
            intervals: list[tuple[int, int, str]] = []
            incomplete = False
            part_errors: list[str] = []
            for tensor in header.tensors:
                if tensor.name in names:
                    duplicate.add(tensor.name)
                names.add(tensor.name)
                expected = gguf.expected_tensor_bytes(tensor)
                if expected is None:
                    if tensor.type in AMBIGUOUS_TYPES:
                        ambiguous.add(tensor.type)
                    else:
                        unknown.add(tensor.type)
                    expected = tensor.size
                if tensor.offset < 0 or tensor.offset >= payload:
                    incomplete = True
                elif tensor.offset + expected > payload:
                    incomplete = True
                present_bytes = min(expected, max(0, payload - tensor.offset))
                intervals.append((tensor.offset, tensor.offset + present_bytes, tensor.name))
                item = histogram.setdefault(_type_name(tensor.type), {"tensors": 0, "bytes": 0})
                item["tensors"] += 1
                item["bytes"] += present_bytes
            intervals.sort()
            for (a0, a1, an), (b0, _b1, bn) in zip(intervals, intervals[1:]):
                if b0 < a1:
                    part_errors.append(f"{part.name}: tensors {an} and {bn} overlap")
                    errors.append(part_errors[-1])
            if incomplete:
                incomplete_parts += 1
                expected_size = header.data_start + max(
                    (t.offset + (gguf.expected_tensor_bytes(t) or t.size) for t in header.tensors), default=0
                )
                warnings.append(
                    f"{part.name}: incomplete ({_human(header.file_size)} of about {_human(expected_size)})"
                )
                checks.append(_check(part.name, "fail", "incomplete", "Wait for the download to finish."))
            elif part_errors:
                checks.append(_check(part.name, "warn", "offset warnings"))
            else:
                checks.append(_check(part.name, "ok", f"{len(header.tensors)} tensors, {_human(payload)} payload"))
            if isinstance(split_tensors, int) and len(header.tensors) > split_tensors and split_tensors > 0:
                warnings.append(f"{part.name}: more tensors than split.tensors.count")
        except (OSError, gguf.GGUFError) as exc:
            errors.append(f"{part.name}: {exc}")
            checks.append(_check(part.name, "fail", str(exc)))

    if split_counts and len(split_counts) > 1:
        errors.append(f"Parts disagree about split.count: {sorted(split_counts)}")
    if len(archs) > 1:
        errors.append(f"Parts disagree about the architecture: {sorted(archs)}")
    if duplicate:
        warnings.append(f"{len(duplicate)} tensor name(s) appear in more than one part; the first one wins.")
    if ambiguous:
        warnings.append(
            f"Tensor type ids {sorted(ambiguous)} mean different types in different engine builds; "
            "the byte count comes from the offsets."
        )
    if unknown:
        errors.append(f"Unknown tensor types: {sorted(unknown)}. This build cannot load them.")
        checks.append(_check("tensor types", "fail", f"unknown ids {sorted(unknown)}", "Use an engine build that knows these types."))
    else:
        checks.append(_check("tensor types", "ok", f"{len(histogram)} known types"))
    expected = next((h.meta.get("split.tensors.count") for h in header_by_part if isinstance(h.meta.get("split.tensors.count"), int)), None)
    if isinstance(expected, int):
        ok = total == expected or bool(missing)
        checks.append(_check("tensor count", "ok" if ok else "fail", f"{total} in the files, {expected} expected"))
        if not ok:
            errors.append(f"Header says {expected} tensors; the parts contain {total}.")
    if (model.get("mtp") or {}).get("head") and not (model.get("mtp") or {}).get("vocab"):
        warnings.append("The MTP head is there, but its vocabulary file is missing; MTP stays off.")
    return {
        "ok": not errors and incomplete_parts == 0,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "parts": len(parts),
        "present": len(present),
        "incomplete": incomplete_parts,
        "tensor_count": total,
        "size": sum(p.stat().st_size for p in present),
        "type_histogram": histogram,
        "unknown_types": sorted(unknown),
    }


def _human(n: int | float | None) -> str:
    value = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _engine_info(studio: Any, info: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    try:
        engine = studio.engine_for("auto", info)
    except Exception as exc:  # ApiError and RuntimeError both carry a usable message
        return None, str(getattr(exc, "message", exc))
    detected: dict[str, Any] = {}
    try:
        detected = studio.engines.detect(engine.get("path") or "")
    except Exception:
        detected = {}
    return {**engine, **detected}, ""


def doctor(studio: Any, model_ident: str = "") -> dict[str, Any]:
    """One read-only check of app, model, engine, memory, disk and the plan."""
    model = studio.library.get(model_ident) if model_ident else None
    if not model:
        model = studio.library.get(str(studio.config.get("last_model") or ""))
    info = studio.sysinfo(fresh=True)
    settings = studio.config.data
    preset = str(settings.get("launch", {}).get("preset") or "auto").lower()
    checks: list[dict[str, str]] = [
        _check("FLE Studio", "ok", f"version {__version__}"),
    ]
    verification: dict[str, Any] | None = None
    if not model:
        checks.append(_check("model", "warn", "no model selected", "Choose a model on the Models page."))
    else:
        checks.append(_check("model", "ok" if model.get("complete") else "fail", model.get("name") or model.get("file", ""),
                             "" if model.get("complete") else "Wait for the download to finish."))
        if model.get("error"):
            checks.append(_check("model header", "fail", str(model["error"]), "Inspect or download the model again."))
        verification = verify(model)
        if verification["errors"]:
            for error in verification["errors"][:3]:
                checks.append(_check("model check", "fail", error))
        elif verification["incomplete"]:
            checks.append(_check("model check", "fail", f"{verification['incomplete']} part(s) still incomplete",
                                 "Wait for the download to finish."))
        else:
            checks.append(_check("model check", "ok", f"{verification['tensor_count']} tensors, offsets fit"))
        for warning in verification["warnings"][:3]:
            checks.append(_check("model warning", "warn", warning))

    engine, engine_error = _engine_info(studio, info)
    if engine_error:
        checks.append(_check("engine", "fail", engine_error, "Install an engine on the Engine page."))
    elif engine:
        checks.append(_check("engine", "ok", f"{engine.get('kind', '?')} · {engine.get('version') or engine.get('path', '')}"))
        if not engine.get("ok", True):
            checks.append(_check("engine", "fail", engine.get("problem") or "does not start"))
        if preset not in ("off", "auto") and engine.get("flavor") != "fle":
            checks.append(_check("engine flavor", "fail", "This is not an FLE engine.",
                                 "Choose an FLE engine, or set the preset to Off/Automatic."))
        if preset not in ("off", "auto") and not engine.get("log_verbosity", True):
            checks.append(_check("engine flags", "warn", "No -lv/--log-verbosity; the live log stays empty."))

    nvidia = [g for g in info.get("gpus", []) if g.get("vendor") == "nvidia"]
    if engine and engine.get("flavor") == "fle" and not nvidia:
        checks.append(_check("GPU", "fail", "no NVIDIA GPU found", "Use the upstream CPU engine with the Off preset."))
    elif not nvidia:
        checks.append(_check("GPU", "warn", "no NVIDIA GPU found", "The engine will run on the CPU."))
    elif nvidia:
        free = sum(max(0, int(g.get("memory_total", 0)) - int(g.get("memory_used", 0))) for g in nvidia)
        checks.append(_check("GPU", "ok", f"{len(nvidia)} GPU(s), {_human(free)} free"))
        if model and model.get("info"):
            model_info = model["info"]
            kv = float(model_info.get("kv_bytes_per_token_q8") or 0) * int(settings["launch"].get("context") or 32768)
            need = float(model_info.get("gpu_bytes") or 0) + kv + launch.ENGINE_OVERHEAD_MIB * 1024 * 1024
            if need > free:
                checks.append(_check("GPU memory", "warn", f"about {_human(need)} needed, {_human(free)} free",
                                     "Stop other GPU work, lower the context, or try the Balanced preset."))
            else:
                checks.append(_check("GPU memory", "ok", f"{_human(free - need)} left for the expert cache"))
    mem = info.get("memory") or {}
    if model and model.get("info"):
        expert = float(model["info"].get("expert_bytes") or 0)
        available = int(mem.get("available") or 0)
        if expert > available:
            checks.append(_check("RAM", "warn", f"experts {_human(expert)}, {_human(available)} available",
                                 "Close other programs, or use the NVMe preset."))
        else:
            checks.append(_check("RAM", "ok", f"experts {_human(expert)}, {_human(available)} available"))
    try:
        free_disk = shutil.disk_usage(Path(model["path"]).parent).free if model else 0
    except OSError:
        free_disk = 0
    if model and free_disk and free_disk < int(model.get("size") or 0):
        checks.append(_check("disk", "warn", f"{_human(free_disk)} free for a {_human(model.get('size'))} model"))

    plan = None
    plan_error = ""
    if model and not model.get("complete"):
        plan_error = "The model is not completely downloaded yet"
        checks.append(_check("plan", "fail", plan_error, "Wait for the download to finish."))
    elif model:
        try:
            plan = studio.make_plan(str(model["id"]), "auto")
            checks.append(_check("plan", "ok", f"{plan.profile}, {len(plan.args)} arguments, port {plan.port}"))
            for note in plan.notes[:2]:
                checks.append(_check("plan note", "ok", note))
        except Exception as exc:
            plan_error = str(getattr(exc, "message", exc))
            checks.append(_check("plan", "fail", plan_error, "Fix the model or engine, then check again."))

    status = _worst(checks)
    return {
        "ok": status != "fail",
        "status": status,
        "app": "fle-studio",
        "version": __version__,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "preset": preset,
        "config_file": str(studio.config.path),
        "model": model,
        "verify": verification,
        "engine": engine,
        "engine_error": engine_error,
        "plan": plan.public() if plan else None,
        "plan_error": plan_error,
        "system": info,
        "checks": checks,
    }


def report(studio: Any, model_ident: str = "") -> dict[str, Any]:
    data = doctor(studio, model_ident)
    data["settings"] = studio.config.public()
    data["engine_status"] = studio.engine.status()
    try:
        data["log"] = studio.engine.log_since(0, limit=300)
    except Exception:
        data["log"] = {"lines": [], "seq": 0}
    data["downloads"] = [
        {k: d.get(k) for k in ("id", "title", "state", "done", "total", "error")}
        for d in studio.downloads.list()
    ]
    return _redact(data)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            upper = str(key).upper()
            if any(word in upper for word in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                out[key] = "<redacted>"
            elif upper.endswith("_KEY") or upper == "KEY":
                out[key] = "<redacted>"
            else:
                out[key] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def report_json(studio: Any, model_ident: str = "") -> str:
    return json.dumps(report(studio, model_ident), ensure_ascii=False, indent=2, default=str)
