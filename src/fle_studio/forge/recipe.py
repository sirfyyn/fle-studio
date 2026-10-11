"""The Forge recipe: one JSON document that says how a model is built, so that anyone can build it again.

Schema version 1 (see docs/FORGE.md for the full description):

    {"schema": 1, "kind": "fle-forge-recipe", "name": ..., "mode": "hybrid" | "quantize",
     "sources": {"B": {"file": ..., "path": ..., "role": "base" | "donor" | "reference", "size": ..., "sha256": ...}},
     "hybrid": {...}, "quantize": {...}, "imatrix": {...}, "reap": {...}, "verify": {...}}

A recipe that is shared carries file names but no folders (`portable`); the person who loads it points each source
at a file of their own (`bind`). A recipe never contains a command line: Forge builds every command itself from the
fields below, so a recipe from someone else cannot make Studio run anything but its own steps.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any

from .. import __version__
from . import ggufio

SCHEMA = 1
KIND = "fle-forge-recipe"
MODES = ("hybrid", "quantize")
ROLES = ("base", "donor", "reference")
REAP_SCORES = ("reap", "freq", "mass", "man", "energy")
PATTERN_OK = re.compile(r"^[\w.*+?|()\[\]\\-]{1,120}$")  # what a tensor-name pattern may be made of
SOURCE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
NAME_OK = re.compile(r"[^\w.-]+")


class RecipeError(ValueError):
    pass


DEFAULTS: dict[str, Any] = {
    "schema": SCHEMA,
    "kind": KIND,
    "name": "my-quant",
    "created": "",
    "studio": "",
    "notes": "",
    "mode": "hybrid",
    "sources": {},
    "hybrid": {
        "base": "",  # source id whose metadata and non-expert tensors the result keeps
        "layers": {},  # {"12": "K"}: expert layer -> source id (layers not listed come from the base)
        "rules": [],  # [{"source": "K", "pattern": "attn", "layers": "0-11"}]: further tensors taken from a donor
        "budget_gb": 0.0,  # the size the layer choice was made for (informational once `layers` is filled)
        "objective": "bpw",  # bpw (stand-in error from bits per weight) | errors (a table of measured errors)
        "errors": {},  # {"K": {"12": 0.031}}: measured error per source and layer, used when objective is errors
    },
    "quantize": {
        "source": "",  # source id of the full-precision model (BF16/F16/F32) or a larger quant
        "type": "iq4_nl",  # the default type of llama-quantize
        "rules": [],  # [{"pattern": "ffn_.*_exps", "type": "iq3_s"}] -> --tensor-type PATTERN=TYPE
        "output_type": "",
        "embedding_type": "",
        "allow_requantize": False,
        "imatrix": True,  # compute an importance matrix first (or use `imatrix.file`)
    },
    "imatrix": {"corpus": "", "file": "", "chunks": 200, "context": 2048, "gpu_layers": 0, "threads": 0},
    "reap": {"enabled": False, "keep": 0, "score": "reap", "table": "", "chunks": 256, "gpu_layers": 0, "threads": 0},
    "verify": {
        "enabled": True,
        "reference": "",
        "text": "",
        "context": 2048,
        "chunks": 20,
        "gpu_layers": 0,
        "threads": 0,
        "base_logits": "",
    },
}


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if key not in base:
            continue  # unknown keys are dropped, like in the settings
        if isinstance(base[key], dict) and key not in ("sources", "layers", "errors"):
            out[key] = _merge(base[key], value) if isinstance(value, dict) else out[key]
        else:
            out[key] = copy.deepcopy(value)
    return out


def clean_name(name: str) -> str:
    return NAME_OK.sub("-", str(name or "").strip()).strip("-.")[:80] or "my-quant"


def _num(value: Any, what: str, low: float, high: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise RecipeError(f"{what} must be a number") from None
    if not low <= v <= high:
        raise RecipeError(f"{what} must be between {low:g} and {high:g}")
    return v


def parse_layer_set(text: str) -> set[int]:
    """'0-3,7' -> {0, 1, 2, 3, 7}"""
    out: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            a, b = int(lo), int(hi or lo)
        except ValueError:
            raise RecipeError(f"not a layer list: {text!r}") from None
        if a < 0 or b < a or b > 4096:
            raise RecipeError(f"not a layer list: {text!r}")
        out.update(range(a, b + 1))
    return out


def normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults, drop unknown keys and check every value. Raises RecipeError with a readable reason."""
    if not isinstance(data, dict):
        raise RecipeError("a recipe is a JSON object")
    if data.get("kind", KIND) != KIND:
        raise RecipeError("this is not a Forge recipe")
    schema = data.get("schema", SCHEMA)
    if schema != SCHEMA:
        raise RecipeError(f"recipe schema {schema} is not supported (this Studio reads schema {SCHEMA})")
    r = _merge(DEFAULTS, data)
    r["name"] = clean_name(r["name"])
    r["notes"] = str(r["notes"])[:2000]
    if r["mode"] not in MODES:
        raise RecipeError(f"mode must be one of {', '.join(MODES)}")

    sources: dict[str, dict[str, Any]] = {}
    for sid, src in (r["sources"] or {}).items():
        if not SOURCE_ID.match(str(sid)):
            raise RecipeError(f"source id {sid!r} must be a letter followed by letters, digits or _ (16 at most)")
        if not isinstance(src, dict):
            raise RecipeError(f"source {sid} must be an object")
        role = str(src.get("role") or "donor")
        if role not in ROLES:
            raise RecipeError(f"source {sid}: role must be one of {', '.join(ROLES)}")
        sha = str(src.get("sha256") or "").lower()
        if sha and not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise RecipeError(f"source {sid}: sha256 is not a hex digest")
        path = str(src.get("path") or "")
        sources[str(sid)] = {
            "file": str(src.get("file") or path.replace("\\", "/").rsplit("/", 1)[-1]),
            "path": path,
            "role": role,
            "size": int(src.get("size") or 0),
            "sha256": sha,
            "origin": str(src.get("origin") or "")[:300],
        }
    r["sources"] = sources

    h = r["hybrid"]
    h["budget_gb"] = _num(h["budget_gb"], "hybrid.budget_gb", 0, 100000)
    if h["objective"] not in ("bpw", "errors"):
        raise RecipeError("hybrid.objective must be bpw or errors")
    layers: dict[str, str] = {}
    for layer, sid in (h["layers"] or {}).items():
        try:
            n = int(layer)
        except (TypeError, ValueError):
            raise RecipeError(f"hybrid.layers: {layer!r} is not a layer number") from None
        if n < 0 or n > 4096:
            raise RecipeError(f"hybrid.layers: layer {n} is out of range")
        layers[str(n)] = str(sid)
    h["layers"] = layers
    rules = []
    for rule in h["rules"] or []:
        pattern = str((rule or {}).get("pattern") or "")
        if not PATTERN_OK.match(pattern):
            raise RecipeError(f"hybrid.rules: bad pattern {pattern!r}")
        layer_text = str(rule.get("layers") or "")
        parse_layer_set(layer_text)
        rules.append({"source": str(rule.get("source") or ""), "pattern": pattern, "layers": layer_text})
    h["rules"] = rules
    errors: dict[str, dict[str, float]] = {}
    for sid, table in (h["errors"] or {}).items():
        errors[str(sid)] = {str(int(k)): float(v) for k, v in (table or {}).items()}
    h["errors"] = errors

    q = r["quantize"]
    for key in ("type", "output_type", "embedding_type"):
        q[key] = str(q[key] or "").lower()
        allowed = ggufio.QUANTIZE_FTYPES if key == "type" else ggufio.QUANTIZE_TARGETS
        if q[key] and q[key] not in allowed:
            raise RecipeError(f"quantize.{key}: unknown type {q[key]!r}")
    if not q["type"]:
        raise RecipeError("quantize.type is required")
    qrules = []
    for rule in q["rules"] or []:
        pattern, typ = str((rule or {}).get("pattern") or ""), str(rule.get("type") or "").lower()
        if not PATTERN_OK.match(pattern):
            raise RecipeError(f"quantize.rules: bad pattern {pattern!r}")
        if typ not in ggufio.QUANTIZE_TARGETS:
            raise RecipeError(f"quantize.rules: unknown type {typ!r}")
        qrules.append({"pattern": pattern, "type": typ})
    q["rules"] = qrules
    q["allow_requantize"] = bool(q["allow_requantize"])
    q["imatrix"] = bool(q["imatrix"])

    im = r["imatrix"]
    im["chunks"] = int(_num(im["chunks"], "imatrix.chunks", 0, 1_000_000))
    im["context"] = int(_num(im["context"], "imatrix.context", 64, 1_048_576))
    im["gpu_layers"] = int(_num(im["gpu_layers"], "imatrix.gpu_layers", 0, 1000))
    im["threads"] = int(_num(im["threads"], "imatrix.threads", 0, 1024))

    reap = r["reap"]
    reap["enabled"] = bool(reap["enabled"])
    reap["keep"] = int(_num(reap["keep"], "reap.keep", 0, 100000))
    if reap["score"] not in REAP_SCORES:
        raise RecipeError(f"reap.score must be one of {', '.join(REAP_SCORES)}")
    reap["chunks"] = int(_num(reap["chunks"], "reap.chunks", 0, 1_000_000))
    reap["gpu_layers"] = int(_num(reap["gpu_layers"], "reap.gpu_layers", 0, 1000))
    reap["threads"] = int(_num(reap["threads"], "reap.threads", 0, 1024))
    if reap["enabled"] and reap["keep"] < 1:
        raise RecipeError("reap.keep must be at least 1 when pruning is on")

    v = r["verify"]
    v["enabled"] = bool(v["enabled"])
    v["context"] = int(_num(v["context"], "verify.context", 64, 1_048_576))
    v["chunks"] = int(_num(v["chunks"], "verify.chunks", 0, 1_000_000))
    v["gpu_layers"] = int(_num(v["gpu_layers"], "verify.gpu_layers", 0, 1000))
    v["threads"] = int(_num(v["threads"], "verify.threads", 0, 1024))
    return r


def validate_sources(r: dict[str, Any]) -> list[str]:
    """Problems that make the recipe unbuildable as it stands (missing source ids, wrong roles). Empty list = fine."""
    problems: list[str] = []
    src = r["sources"]

    def need(sid: str, what: str) -> None:
        if not sid:
            problems.append(f"{what} is not set")
        elif sid not in src:
            problems.append(f"{what} names the source {sid!r}, which the recipe does not list")

    if r["mode"] == "hybrid":
        h = r["hybrid"]
        need(h["base"], "hybrid.base")
        for layer, sid in h["layers"].items():
            if sid not in src:
                problems.append(f"layer {layer} comes from {sid!r}, which the recipe does not list")
        for rule in h["rules"]:
            need(rule["source"], "a hybrid rule's source")
    else:
        need(r["quantize"]["source"], "quantize.source")
    if r["verify"]["enabled"]:
        need(r["verify"]["reference"], "verify.reference")
    for sid, s in src.items():
        if not s["path"]:
            problems.append(f"source {sid} ({s['file'] or 'no file name'}) has no file on this computer yet")
    if r["reap"]["enabled"] and r["reap"]["keep"] < 1:
        problems.append("reap.keep is not set")
    return list(dict.fromkeys(problems))


def new(name: str, mode: str = "hybrid") -> dict[str, Any]:
    r = normalize({"name": name, "mode": mode})
    r["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    r["studio"] = __version__
    return r


def portable(r: dict[str, Any]) -> dict[str, Any]:
    """The recipe as it is shared: no folders, no corpus or table paths of this computer, no local sizes check."""
    out = copy.deepcopy(normalize(r))
    for s in out["sources"].values():
        s["path"] = ""
    out["imatrix"]["corpus"] = (
        out["imatrix"]["corpus"].replace("\\", "/").rsplit("/", 1)[-1] if out["imatrix"]["corpus"] else ""
    )
    for section, key in (("imatrix", "file"), ("reap", "table"), ("verify", "text"), ("verify", "base_logits")):
        value = out[section][key]
        out[section][key] = value.replace("\\", "/").rsplit("/", 1)[-1] if value else ""
    out["studio"] = out["studio"] or __version__
    return out


def bind(r: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    """Point sources at files of this computer: {source id: path}."""
    out = copy.deepcopy(normalize(r))
    for sid, path in files.items():
        if sid in out["sources"]:
            out["sources"][sid]["path"] = str(path)
    return out


def dumps(r: dict[str, Any]) -> str:
    return json.dumps(r, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def loads(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RecipeError(f"not valid JSON: {exc}") from None
    return normalize(data)
