"""The recipe calculator: which source supplies which expert layer, and what the result needs on your machine.

A hybrid keeps the metadata and every non-expert tensor of one base file and takes whole expert layers (gate, up and
down together, or the fused gate_up) from the base or from other quants of the same model. Which layer comes from
which source is a multiple-choice knapsack: one pick per layer, the sum of the layer sizes within the budget, the
sum of the predicted errors as small as possible. It is solved exactly by dynamic programming over the bytes (in
steps that never round a size down, so the result never exceeds the budget).

The error per layer and source is, in order of preference, a table you measured (`errors`) or a stand-in computed
from the bits per weight, 4^-bpw, which is the usual rate-distortion slope. The stand-in spreads the bits evenly; it
cannot know that some layers matter more. Measure the result (KL divergence) before trusting either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .. import gguf
from . import ggufio

EXPERT_LAYER = re.compile(r"^blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight$")
GB = 1_000_000_000
GIB = 1 << 30
# Tensors llama-quantize leaves in their type (routers, norms, positional tables, state-space parameters).
NEVER_QUANTIZED = (
    "_norm.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_inp_shexp.weight",
    "ssm_conv1d.weight",
    "pos_embd",
    "token_types",
    "attn_rel_b.weight",
)
MAX_STEPS = 20000  # resolution of the knapsack: the expert budget is cut into at most this many steps


class PlanError(ValueError):
    pass


def _padded(n: int, alignment: int) -> int:
    return -(-n // alignment) * alignment


@dataclass
class Source:
    """What the calculator needs to know about one file (or split file) of a model."""

    id: str
    model: ggufio.Model
    layer_bytes: dict[int, int] = field(default_factory=dict)  # padded bytes of the expert tensors per layer
    layer_elems: dict[int, int] = field(default_factory=dict)
    layer_types: dict[int, dict[str, str]] = field(default_factory=dict)
    layer_shapes: dict[int, tuple[tuple[str, tuple[int, ...]], ...]] = field(default_factory=dict)

    @classmethod
    def load(cls, sid: str, path: str) -> Source:
        return cls.from_model(sid, ggufio.open_model(path))

    @classmethod
    def from_model(cls, sid: str, model: ggufio.Model) -> Source:
        src = cls(sid, model)
        a = model.alignment
        for name in model.order:
            m = EXPERT_LAYER.match(name)
            if not m:
                continue
            layer, kind = int(m.group(1)), m.group(2)
            t = model.tensors[name]
            src.layer_bytes[layer] = src.layer_bytes.get(layer, 0) + _padded(t.nbytes, a)
            src.layer_elems[layer] = src.layer_elems.get(layer, 0) + t.n_elements
            src.layer_types.setdefault(layer, {})[kind] = t.type_name
            src.layer_shapes[layer] = (*src.layer_shapes.get(layer, ()), (name, t.shape))
        return src

    @property
    def layers(self) -> list[int]:
        return sorted(self.layer_bytes)

    @property
    def rest_bytes(self) -> int:
        a = self.model.alignment
        return sum(_padded(t.nbytes, a) for n, t in self.model.tensors.items() if not EXPERT_LAYER.match(n))

    def bpw(self, layer: int) -> float:
        return self.layer_bytes[layer] * 8 / max(1, self.layer_elems[layer])

    def describe(self) -> dict[str, Any]:
        m = self.model
        groups: dict[str, list[int]] = {}
        for layer in self.layers:
            combo = " / ".join(f"{k} {v}" for k, v in sorted(self.layer_types[layer].items()))
            groups.setdefault(combo, []).append(layer)
        return {
            "id": self.id,
            "path": m.files[0].path,
            "parts": len(m.files),
            "size": m.total_bytes,
            "arch": m.arch,
            "name": str(m.meta.get("general.name") or ""),
            "layers": m.number("block_count"),
            "experts": m.number("expert_count"),
            "experts_used": m.number("expert_used_count"),
            "expert_layers": len(self.layer_bytes),
            "expert_bytes": sum(self.layer_bytes.values()),
            "rest_bytes": self.rest_bytes,
            "types": [{"types": k, "layers": v} for k, v in sorted(groups.items(), key=lambda kv: kv[1][0])],
            "bpw": round(sum(self.layer_bytes.values()) * 8 / max(1, sum(self.layer_elems.values())), 3)
            if self.layer_bytes
            else None,
        }


def check_compatible(base: Source, other: Source) -> None:
    """A donor can replace an expert layer only if the layer has the same tensors with the same shapes."""
    if base.model.arch != other.model.arch:
        raise PlanError(f"{other.id} is a {other.model.arch!r} model, {base.id} is {base.model.arch!r}")
    if base.layers != other.layers:
        raise PlanError(f"{other.id} has expert layers {other.layers[:3]}..., {base.id} has {base.layers[:3]}...")
    for layer in base.layers:
        if base.layer_shapes[layer] != other.layer_shapes[layer]:
            raise PlanError(
                f"layer {layer}: the expert tensors of {other.id} and {base.id} differ in name or shape; "
                "they are not quants of the same model"
            )


@dataclass
class Selection:
    layers: dict[int, str]  # layer -> source id
    feasible: bool
    bytes: int  # data bytes of the result (padded tensors), without the header
    min_bytes: int
    max_bytes: int
    error: float
    error_base: float
    note: str = ""

    def public(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for sid in self.layers.values():
            counts[sid] = counts.get(sid, 0) + 1
        return {
            "layers": {str(k): v for k, v in sorted(self.layers.items())},
            "feasible": self.feasible,
            "bytes": self.bytes,
            "min_bytes": self.min_bytes,
            "max_bytes": self.max_bytes,
            "error": self.error,
            "error_base": self.error_base,
            "counts": counts,
            "note": self.note,
        }


def layer_errors(sources: dict[str, Source], errors: dict[str, dict[str, float]] | None) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for sid, src in sources.items():
        table = (errors or {}).get(sid)
        if errors:
            if table is None:
                raise PlanError(f"the error table has no entry for source {sid}")
            missing = [layer for layer in src.layers if str(layer) not in table]
            if missing:
                raise PlanError(f"the error table of {sid} lacks layer(s) {missing[:5]}")
            out[sid] = {layer: float(table[str(layer)]) for layer in src.layers}
        else:
            out[sid] = {layer: 4.0 ** (-src.bpw(layer)) for layer in src.layers}
    return out


def choose_layers(
    sources: dict[str, Source],
    base: str,
    budget_bytes: int,
    errors: dict[str, dict[str, float]] | None = None,
    header_bytes: int = 0,
) -> Selection:
    """Pick the source of every expert layer for the smallest summed error within `budget_bytes` of file size."""
    if base not in sources:
        raise PlanError(f"base source {base!r} is not among the sources")
    b = sources[base]
    for src in sources.values():
        check_compatible(b, src)
    layers = b.layers
    if not layers:
        raise PlanError("the base model has no expert layers (is it a mixture-of-experts model?)")
    err = layer_errors(sources, errors)
    ids = list(sources)
    rest = b.rest_bytes + header_bytes
    cheapest = {layer: min(sources[s].layer_bytes[layer] for s in ids) for layer in layers}
    dearest = {layer: max(sources[s].layer_bytes[layer] for s in ids) for layer in layers}
    min_total, max_total = rest + sum(cheapest.values()), rest + sum(dearest.values())
    base_err = sum(err[base][layer] for layer in layers)
    base_total = rest + sum(b.layer_bytes.values())
    if budget_bytes < min_total:
        pick = {layer: min(ids, key=lambda s, lay=layer: sources[s].layer_bytes[lay]) for layer in layers}
        return Selection(
            pick,
            False,
            min_total,
            min_total,
            max_total,
            sum(err[pick[x]][x] for x in layers),
            base_err,
            f"The budget is below the smallest possible result ({min_total / GB:.2f} GB).",
        )
    avail = budget_bytes - rest - sum(cheapest.values())  # bytes that can be spent above the cheapest pick
    step = max(1, -(-avail // MAX_STEPS)) if avail > 0 else 1
    cap = avail // step
    # state: extra units over the cheapest pick -> (error, back pointer)
    layers_sorted = sorted(layers)
    tables: list[dict[int, tuple[float, int, str]]] = []
    states: dict[int, tuple[float, int, str]] = {0: (0.0, -1, "")}
    for layer in layers_sorted:
        nxt: dict[int, tuple[float, int, str]] = {}
        for used, (e0, _, _) in states.items():
            for sid in ids:
                extra = -(-(sources[sid].layer_bytes[layer] - cheapest[layer]) // step)
                u = used + extra
                if u > cap:
                    continue
                e1 = e0 + err[sid][layer]
                if u not in nxt or e1 < nxt[u][0]:
                    nxt[u] = (e1, used, sid)
        front: dict[int, tuple[float, int, str]] = {}
        best = float("inf")
        for u in sorted(nxt):  # keep only states that are better than every state with fewer bytes
            if nxt[u][0] < best:
                front[u], best = nxt[u], nxt[u][0]
        tables.append(front)
        states = front
    used = min(states, key=lambda u: states[u][0])
    pick: dict[int, str] = {}
    for layer, table in zip(reversed(layers_sorted), reversed(tables), strict=True):
        e, prev, sid = table[used]
        pick[layer] = sid
        used = prev
    total = rest + sum(sources[pick[x]].layer_bytes[x] for x in layers)
    error = sum(err[pick[x]][x] for x in layers)
    note = ""
    if total > base_total:
        note = "The result is larger than the base file."
    return Selection(pick, True, total, min_total, max_total, error, base_err, note)


# ------------------------------------------------------------------------------------------------ result size


def assemble(
    model: ggufio.Model,
    donors: dict[str, ggufio.Model],
    pick: dict[int, str],
    base: str,
    rules: list[tuple[str, re.Pattern[str], set[int] | None]] | None = None,
) -> dict[str, str]:
    """For every tensor of the base model, the source id it is taken from (checked: it must exist and have the
    same shape and type family). Raises PlanError for a donor that lacks a tensor or has another shape."""
    plan: dict[str, str] = {}
    for name in model.order:
        sid = base
        m = EXPERT_LAYER.match(name)
        if m:
            sid = pick.get(int(m.group(1)), base)
        layer_m = re.match(r"^blk\.(\d+)\.", name)
        layer = int(layer_m.group(1)) if layer_m else None
        for rule_source, pattern, layers in rules or []:
            if pattern.search(name) and (layers is None or (layer is not None and layer in layers)):
                if sid != base and rule_source != sid:
                    raise PlanError(f"{name}: claimed by two sources ({sid}, {rule_source})")
                sid = rule_source
        if sid != base:
            donor = donors[sid]
            if name not in donor.tensors:
                raise PlanError(f"source {sid} has no tensor {name}")
            if donor.tensors[name].shape != model.tensors[name].shape:
                raise PlanError(
                    f"{name}: shape {donor.tensors[name].shape} in {sid}, {model.tensors[name].shape} in the base"
                )
        plan[name] = sid
    return plan


def result_tensors(
    model: ggufio.Model, models: dict[str, ggufio.Model], plan: dict[str, str]
) -> list[tuple[str, tuple[int, ...], int, int]]:
    out = []
    for name in model.order:
        t = models[plan[name]].tensors[name]
        out.append((name, t.shape, t.type, t.nbytes))
    return out


def predicted_file_size(model: ggufio.Model, models: dict[str, ggufio.Model], plan: dict[str, str]) -> int:
    """The exact size of the file a build will write, computed from headers only."""
    infos = result_tensors(model, models, plan)
    header, offsets = ggufio.build_header(model.kv, infos, model.alignment)
    _, data = ggufio.layout(infos, model.alignment)
    return len(header) + data


# ------------------------------------------------------------------------------------------------ memory preview


def kv_bytes_per_token(model: ggufio.Model, type_k: str = "q8_0", type_v: str = "q8_0") -> float:
    info = gguf.ModelInfo(
        path="",
        parts=[],
        missing=[],
        size=0,
        arch=model.arch,
        name="",
        context_length=None,
        layers=None,
        experts=None,
        experts_used=None,
        expert_bytes=0,
        input_bytes=0,
        gpu_bytes=0,
        has_chat_template=False,
        file_type=None,
        meta=dict(model.meta),
    )
    return info.kv_bytes_per_token(type_k, type_v)


def memory_preview(
    model: ggufio.Model,
    models: dict[str, ggufio.Model],
    plan: dict[str, str],
    ram_bytes: int,
    vram_bytes: int,
    context: int = 32768,
    type_k: str = "q8_0",
    type_v: str = "q8_0",
) -> dict[str, Any]:
    """What the result needs, split the way llama.cpp places it, and whether it fits the given RAM and VRAM.

    Layout "experts on the CPU" keeps the routed experts and the token tables in RAM and everything else on the GPU
    (the usual way to run a large mixture-of-experts model); "all on the GPU" is only for small models."""
    experts = inputs = other = 0
    for name in model.order:
        t = models[plan[name]].tensors[name]
        if gguf.EXPERT_RE.search(name):
            experts += t.nbytes
        elif name.startswith(gguf.INPUT_PREFIXES):
            inputs += t.nbytes
        else:
            other += t.nbytes
    kv = int(kv_bytes_per_token(model, type_k, type_v) * context)
    runtime_ram, runtime_gpu = 2 * GIB, int(1.5 * GIB)
    split = {"ram": experts + inputs + runtime_ram, "vram": other + kv + runtime_gpu}
    split["gpu_cache"] = max(0, vram_bytes - split["vram"]) if vram_bytes else 0
    split["fits"] = split["ram"] <= ram_bytes and (not vram_bytes or split["vram"] <= vram_bytes)
    on_gpu = {"ram": inputs + runtime_ram, "vram": experts + other + kv + runtime_gpu}
    on_gpu["fits"] = bool(vram_bytes) and on_gpu["ram"] <= ram_bytes and on_gpu["vram"] <= vram_bytes
    unified = {"ram": experts + inputs + other + kv + runtime_ram, "vram": 0}
    unified["fits"] = unified["ram"] <= ram_bytes
    if vram_bytes == 0:
        verdict = "cpu_only" if unified["fits"] else "too_big"
    elif on_gpu["fits"]:
        verdict = "all_gpu"
    elif split["fits"]:
        verdict = "experts_cpu"
    elif unified["fits"]:
        verdict = "cpu_only"
    else:
        verdict = "too_big"
    return {
        "experts": experts,
        "inputs": inputs,
        "dense": other,
        "kv": kv,
        "context": context,
        "experts_on_cpu": split,
        "all_gpu": on_gpu,
        "cpu_only": unified,
        "verdict": verdict,
        "ram_bytes": ram_bytes,
        "vram_bytes": vram_bytes,
    }


def estimate_quantize(
    model: ggufio.Model, default_type: str, rules: list[dict[str, str]], output_type: str = "", embedding_type: str = ""
) -> dict[str, Any]:
    """An estimate of what llama-quantize will write: every tensor with at least two dimensions gets the first rule
    that matches its name, else the default type; vectors (norms, biases) keep their type. A row that is not a whole
    number of blocks of the new type keeps the old type here, as llama-quantize falls back in that case too (to a
    type it picks itself, so the real file can differ by a little). Marked as an estimate everywhere it is shown."""
    infos = []
    changed = kept = 0
    for name in model.order:
        t = model.tensors[name]
        want: str | None = None
        if len(t.shape) >= 2 and name.endswith(".weight") and not any(x in name for x in NEVER_QUANTIZED):
            if name == "output.weight" and output_type:
                want = output_type
            elif name == "token_embd.weight" and embedding_type:
                want = embedding_type
            else:
                want = next((r["type"] for r in rules if re.search(r["pattern"], name)), default_type)
        tid = ggufio.type_id(want) if want else None
        nbytes = ggufio.tensor_nbytes(tid, t.shape) if tid is not None else None
        if tid is None or nbytes is None:
            infos.append((name, t.shape, t.type, t.nbytes))
            kept += 1
        else:
            infos.append((name, t.shape, tid, nbytes))
            changed += 1
    header, _ = ggufio.build_header(model.kv, infos, model.alignment)
    size = len(header) + ggufio.layout(infos, model.alignment)[1]
    params = sum(t.n_elements for t in model.tensors.values())
    return {
        "estimate_bytes": size,
        "bpw": round(size * 8 / max(params, 1), 3),
        "tensors_changed": changed,
        "tensors_kept": kept,
        "estimate": True,
    }


def parse_rules(rules: list[dict[str, Any]]) -> list[tuple[str, re.Pattern[str], set[int] | None]]:
    from .recipe import parse_layer_set

    out = []
    for rule in rules:
        layers = parse_layer_set(rule["layers"]) if rule.get("layers") else None
        try:
            out.append((rule["source"], re.compile(rule["pattern"]), layers))
        except re.error as exc:
            raise PlanError(f"bad pattern {rule['pattern']!r}: {exc}") from None
    return out
