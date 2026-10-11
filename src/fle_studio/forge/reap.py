"""Expert pruning in the style of REAP: keep the K most useful routed experts of every layer, drop the rest.

Input is a table of router statistics per layer and expert. The FLE engine's `llama-fle-reap` writes it as a TSV
(`layer expert count sum_gate sum_saliency sum_norm`); without that tool a table can be derived from an importance
matrix file, which has the routing counts and the input energy of every expert (a weaker signal, see docs/FORGE.md).

Scores (higher is kept):
  reap    sum_saliency / count   mean router-weighted expert output norm over the tokens routed to the expert (REAP)
  man     sum_norm / count       the same without the router weight
  freq    count                  how often the expert was routed
  mass    sum_gate               summed router weight
  energy  sum_saliency           raw saliency (for tables derived from an importance matrix: input energy)

Pruning copies bytes. The expert axis is the slowest axis of every per-expert tensor, so the experts to keep are
contiguous slices of the source tensor, whatever its quantisation. The router (`ffn_gate_inp`, `exp_probs_b`) loses
the rows of the dropped experts, and `<arch>.expert_count` becomes K. The kept ids go to a manifest next to the file.

Limits: models with grouped routing (expert groups) are refused; the active experts per token must not exceed K;
nothing is re-trained, so the quality of the pruned model has to be measured (KL divergence) like any other.
"""

from __future__ import annotations

import csv
import json
import os
import re
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import build, ggufio

Entry = tuple[int, float, float, float]  # count, sum_gate, sum_saliency, sum_norm
Table = dict[int, dict[int, Entry]]

PER_EXPERT = re.compile(
    r"^blk\.(\d+)\.(ffn_(gate|up|down|gate_up)_exps(\.(weight|bias|scale))?|ffn_(gate|up|down)_exps_s.*|"
    r"ffn_gate_inp(\.weight)?|exp_probs_b(\.bias)?)$"
)
SCORES = ("reap", "freq", "mass", "man", "energy")


class ReapError(ValueError):
    pass


def load_table(path: str | os.PathLike[str]) -> Table:
    table: Table = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        need = {"layer", "expert", "count"}
        if not reader.fieldnames or not need <= set(reader.fieldnames):
            raise ReapError("the table needs the columns layer, expert and count (tab separated)")
        for row in reader:
            table.setdefault(int(row["layer"]), {})[int(row["expert"])] = (
                int(float(row["count"])),
                float(row.get("sum_gate") or 0.0),
                float(row.get("sum_saliency") or 0.0),
                float(row.get("sum_norm") or 0.0),
            )
    if not table:
        raise ReapError("the table is empty")
    return table


def write_table(table: Table, path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("layer\texpert\tcount\tsum_gate\tsum_saliency\tsum_norm\n")
        for layer in sorted(table):
            for expert in sorted(table[layer]):
                c, g, s, n = table[layer][expert]
                f.write(f"{layer}\t{expert}\t{c}\t{g!r}\t{s!r}\t{n!r}\n")


def table_from_imatrix(path: str | os.PathLike[str]) -> Table:
    """Router counts and input energy per expert from an importance-matrix GGUF (the newer file format)."""
    g = ggufio.read_gguf(path)
    counts: dict[int, list[float]] = {}
    energy: dict[int, list[float]] = {}
    with open(path, "rb") as f:
        for t in g.tensors:
            m = re.match(r"^blk\.(\d+)\.ffn_gate_exps\.weight\.(counts|in_sum2)$", t.name)
            if not m or ggufio.type_name(t.type) != "F32":
                continue
            f.seek(g.data_start + t.offset)
            vals = list(struct.unpack(f"<{t.n_elements}f", f.read(t.n_elements * 4)))
            (counts if m.group(2) == "counts" else energy)[int(m.group(1))] = vals
    if not counts:
        raise ReapError(
            "the importance matrix has no per-expert counts (it must be a GGUF written by a recent "
            "llama-imatrix, not the old .dat format)"
        )
    table: Table = {}
    for layer, c in counts.items():
        e = energy.get(layer)
        n_expert = len(c)
        per = len(e) // n_expert if e and len(e) % n_expert == 0 else 0
        table[layer] = {}
        for x in range(n_expert):
            en = sum(e[x * per : (x + 1) * per]) if per else float(c[x])
            table[layer][x] = (int(c[x]), float(c[x]), en, en)
    return table


def score(entry: Entry, kind: str) -> float:
    count, gate, sal, norm = entry
    if kind == "freq":
        return float(count)
    if kind == "mass":
        return gate
    if kind == "man":
        return norm / count if count else 0.0
    if kind == "energy":
        return sal
    return sal / count if count else 0.0


def select(table: Table, layers: list[int], n_expert: int, keep: int, kind: str) -> dict[int, list[int]]:
    """Per layer the K best experts, in their original order. Never-routed experts score 0 and go first."""
    if kind not in SCORES:
        raise ReapError(f"unknown score {kind!r}")
    missing = [layer for layer in layers if layer not in table]
    if missing:
        raise ReapError(f"the table has no data for layer(s) {missing[:8]}; run the router statistics on this model")
    chosen: dict[int, list[int]] = {}
    for layer in layers:
        stats = table[layer]
        ranked = sorted(
            range(n_expert), key=lambda e: (score(stats.get(e, (0, 0.0, 0.0, 0.0)), kind), -e), reverse=True
        )
        chosen[layer] = sorted(ranked[:keep])
    return chosen


def _runs(ids: list[int]) -> list[tuple[int, int]]:
    """[2, 3, 4, 9] -> [(2, 3), (9, 1)]: first index and length of every run of consecutive ids."""
    out: list[tuple[int, int]] = []
    for i in ids:
        if out and out[-1][0] + out[-1][1] == i:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((i, 1))
    return out


def plan_prune(
    model: ggufio.Model, table: Table, keep: int, kind: str
) -> tuple[list[build.OutTensor], list[ggufio.KV], dict[str, Any]]:
    arch = model.arch
    key = f"{arch}.expert_count"
    n_expert = model.number("expert_count")
    if not n_expert:
        raise ReapError("this is not a mixture-of-experts model (no expert_count)")
    if (model.number("expert_group_count") or 1) > 1:
        raise ReapError("models with grouped expert routing are not supported")
    used = model.number("expert_used_count") or 0
    if not 0 < keep < n_expert:
        raise ReapError(f"keep must be between 1 and {n_expert - 1}")
    if keep < used:
        raise ReapError(f"the model activates {used} experts per token; keep at least that many")
    layers = sorted({int(m.group(1)) for n in model.order if (m := PER_EXPERT.match(n))})
    if not layers:
        raise ReapError("no expert tensors found")
    chosen = select(table, layers, n_expert, keep, kind)
    tensors = []
    for name in model.order:
        t = model.tensors[name]
        path, offset, nbytes = model.location(name)
        m = PER_EXPERT.match(name)
        if not m:
            tensors.append(build.OutTensor(name, t.shape, t.type, t.nbytes, [(path, offset, nbytes)]))
            continue
        if t.shape[-1] != n_expert or nbytes % n_expert:
            raise ReapError(f"{name}: expected {n_expert} experts on the last axis, shape is {list(t.shape)}")
        ids = chosen[int(m.group(1))]
        each = nbytes // n_expert
        segments = [(path, offset + first * each, count * each) for first, count in _runs(ids)]
        tensors.append(build.OutTensor(name, (*t.shape[:-1], keep), t.type, each * keep, segments))
    kv = ggufio.with_value(model.kv, key, keep)
    kept_mass = []
    for layer in layers:
        total = sum(v[1] for v in table[layer].values()) or 1.0
        kept_mass.append(sum(table[layer].get(e, (0, 0.0, 0.0, 0.0))[1] for e in chosen[layer]) / total)
    info = {
        "n_expert": n_expert,
        "keep": keep,
        "score": kind,
        "layers": {str(k): v for k, v in chosen.items()},
        "routing_mass_kept": {
            "mean": sum(kept_mass) / len(kept_mass),
            "min": min(kept_mass),
            "min_layer": layers[kept_mass.index(min(kept_mass))],
        },
    }
    return tensors, kv, info


def prune(
    model_path: str,
    out: str,
    table: Table,
    keep: int,
    kind: str = "reap",
    progress: Callable[[float, str], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    model = ggufio.open_model(model_path)
    tensors, kv, info = plan_prune(model, table, keep, kind)
    rm = info["routing_mass_kept"]
    if log:
        log(
            f"{model.number('expert_count')} -> {keep} experts in {len(info['layers'])} layers, score {kind}; "
            f"router mass kept: mean {rm['mean']:.4f}, lowest {rm['min']:.4f} (layer {rm['min_layer']})"
        )
    result = build.write_model(out, kv, tensors, model.alignment, progress, cancelled, log)
    manifest = {"model": Path(model_path).name, **info, "sha256": result["sha256"]}
    Path(str(out) + ".reap.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    result.update({"keep": keep, "n_expert": info["n_expert"], "routing_mass_kept": rm})
    return result


def predicted_prune_size(model: ggufio.Model, keep: int) -> int:
    """Size of the pruned file from the headers alone (every expert tensor shrinks to keep / n_expert)."""
    n_expert = model.number("expert_count") or 0
    infos = []
    for name in model.order:
        t = model.tensors[name]
        if PER_EXPERT.match(name) and n_expert and t.shape[-1] == n_expert:
            infos.append((name, (*t.shape[:-1], keep), t.type, t.nbytes // n_expert * keep))
        else:
            infos.append((name, t.shape, t.type, t.nbytes))
    header, _ = ggufio.build_header(model.kv, infos, model.alignment)
    return len(header) + ggufio.layout(infos, model.alignment)[1]


__all__ = [
    "SCORES",
    "ReapError",
    "load_table",
    "plan_prune",
    "predicted_prune_size",
    "prune",
    "select",
    "table_from_imatrix",
    "write_table",
]
