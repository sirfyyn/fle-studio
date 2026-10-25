from __future__ import annotations

import json
from pathlib import Path

import pytest
from forge_synth import make_imatrix, make_model

from fle_studio_forge import ggufio, reap

LAYERS, EXPERTS = 3, 8


def table_for(counts_by_layer: dict[int, list[int]]) -> reap.Table:
    t: reap.Table = {}
    for layer, counts in counts_by_layer.items():
        t[layer] = {e: (c, float(c), float(c) * (1 + e % 2), float(c)) for e, c in enumerate(counts)}
    return t


COUNTS = {
    0: [10, 90, 5, 80, 70, 1, 60, 40],
    1: [100, 1, 2, 3, 4, 95, 90, 5],
    2: [0, 0, 50, 51, 52, 53, 0, 1],
}


@pytest.fixture
def model(tmp_path: Path) -> str:
    return str(
        make_model(
            tmp_path / "m.gguf",
            seed=11,
            layers=LAYERS,
            experts=EXPERTS,
            used=2,
            types=[("Q4_0", "Q5_0", "Q8_0")] * LAYERS,
        )
    )


def slice_of(path: str, name: str, ids: list[int], n_expert: int = EXPERTS) -> bytes:
    m = ggufio.open_model(path)
    p, off, size = m.location(name)
    each = size // n_expert
    with open(p, "rb") as f:
        out = b""
        for i in ids:
            f.seek(off + i * each)
            out += f.read(each)
    return out


def test_selection_keeps_the_top_experts_in_their_original_order():
    chosen = reap.select(table_for(COUNTS), [0, 1, 2], EXPERTS, 4, "freq")
    assert chosen[0] == [1, 3, 4, 6]
    assert chosen[1] == [0, 5, 6, 7]
    assert chosen[2] == [2, 3, 4, 5]  # never-routed experts go first


def test_the_scores_rank_differently():
    t = table_for({0: [10, 10, 10, 10, 10, 10, 10, 10]})
    t[0][3] = (10, 99.0, 1.0, 1.0)
    assert 3 in reap.select(t, [0], EXPERTS, 1, "mass")[0]
    t[0][5] = (2, 0.1, 100.0, 4.0)  # few tokens but a large mean saliency
    assert reap.select(t, [0], EXPERTS, 1, "reap")[0] == [5]
    assert reap.select(t, [0], EXPERTS, 1, "freq")[0] != [5]
    assert reap.select(t, [0], EXPERTS, 1, "man")[0] == [5]
    with pytest.raises(reap.ReapError):
        reap.select(t, [0], EXPERTS, 1, "nonsense")


def test_a_layer_missing_from_the_table_is_named():
    with pytest.raises(reap.ReapError, match=r"layer\(s\) \[2\]"):
        reap.select(table_for({0: COUNTS[0], 1: COUNTS[1]}), [0, 1, 2], EXPERTS, 4, "freq")


def test_pruning_shrinks_experts_router_and_metadata_consistently(tmp_path: Path, model):
    out = tmp_path / "pruned.gguf"
    result = reap.prune(model, str(out), table_for(COUNTS), 4, "freq")
    src, got = ggufio.open_model(model), ggufio.open_model(str(out))
    chosen = reap.select(table_for(COUNTS), [0, 1, 2], EXPERTS, 4, "freq")

    assert got.number("expert_count") == 4 and got.number("expert_used_count") == 2
    assert got.order == src.order
    before = {k.key: k.raw for k in src.kv}
    after = {k.key: k.raw for k in got.kv}
    assert [k for k in before if before[k] != after[k]] == ["testmoe.expert_count"]  # nothing else in the metadata

    per_expert = (
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_inp.weight",
        "exp_probs_b.bias",
    )
    for name in got.order:
        a, b = src.tensors[name], got.tensors[name]
        layer = int(name.split(".")[1]) if name.startswith("blk.") else None
        if layer is not None and name.split(".", 2)[2] in per_expert:
            assert b.shape == (*a.shape[:-1], 4), name
            assert b.nbytes == a.nbytes // EXPERTS * 4
            assert slice_of(str(out), name, [0, 1, 2, 3], 4) == slice_of(model, name, chosen[layer]), name
        else:
            assert (b.shape, b.type, b.nbytes) == (a.shape, a.type, a.nbytes), name
    assert out.stat().st_size < Path(model).stat().st_size
    assert result["keep"] == 4 and result["size"] == out.stat().st_size
    assert result["size"] == reap.predicted_prune_size(src, 4)

    manifest = json.loads(Path(str(out) + ".reap.json").read_text())
    assert manifest["layers"] == {str(k): v for k, v in chosen.items()}
    assert manifest["routing_mass_kept"]["min_layer"] in (0, 1, 2)


def test_the_router_rows_are_the_kept_experts_rows(tmp_path: Path, model):
    out = tmp_path / "pruned.gguf"
    reap.prune(model, str(out), table_for(COUNTS), 3, "freq")
    name = "blk.1.ffn_gate_inp.weight"  # (hidden, experts) in GGUF order: one row of `hidden` floats per expert
    chosen = reap.select(table_for(COUNTS), [1], EXPERTS, 3, "freq")[1]
    assert slice_of(str(out), name, [0, 1, 2], 3) == slice_of(model, name, chosen)


def test_a_pruned_model_can_be_pruned_again(tmp_path: Path, model):
    once = tmp_path / "once.gguf"
    reap.prune(model, str(once), table_for(COUNTS), 6, "freq")
    t6 = table_for({layer: list(range(6, 0, -1)) for layer in range(LAYERS)})
    twice = tmp_path / "twice.gguf"
    reap.prune(str(once), str(twice), t6, 3, "freq")
    assert ggufio.open_model(str(twice)).number("expert_count") == 3


@pytest.mark.parametrize("keep, message", [(0, "between 1"), (EXPERTS, "between 1"), (1, "activates 2")])
def test_bad_keep_values_are_refused(tmp_path: Path, model, keep, message):
    with pytest.raises(reap.ReapError, match=message):
        reap.prune(model, str(tmp_path / "x.gguf"), table_for(COUNTS), keep, "freq")
    assert not (tmp_path / "x.gguf").exists()


def test_a_model_without_experts_is_refused(tmp_path: Path):
    p = tmp_path / "dense.gguf"
    kv = [ggufio.kv_scalar("general.architecture", ggufio.T_STRING, "testmoe")]
    ggufio.write_simple(p, kv, [("token_embd.weight", (32, 2), ggufio.type_id("Q8_0") or 0, bytes(68 * 2))])
    with pytest.raises(reap.ReapError, match="not a mixture"):
        reap.prune(str(p), str(tmp_path / "x.gguf"), table_for(COUNTS), 2, "freq")


def test_grouped_routing_is_refused(tmp_path: Path):
    from forge_synth import kv_list

    p = tmp_path / "g.gguf"
    ggufio.write_simple(p, kv_list(1, 8, extra=[ggufio.kv_scalar("testmoe.expert_group_count", ggufio.T_U32, 2)]), [])
    with pytest.raises(reap.ReapError, match="grouped"):
        reap.prune(str(p), str(tmp_path / "x.gguf"), table_for(COUNTS), 4, "freq")


def test_a_cancelled_prune_resumes(tmp_path: Path, model):
    ref = tmp_path / "ref.gguf"
    reap.prune(model, str(ref), table_for(COUNTS), 4, "reap")
    out = tmp_path / "out.gguf"
    n = {"i": 0}

    def stop() -> bool:
        n["i"] += 1
        return n["i"] > 25

    with pytest.raises(ggufio.Cancelled):
        reap.prune(model, str(out), table_for(COUNTS), 4, "reap", cancelled=stop)
    reap.prune(model, str(out), table_for(COUNTS), 4, "reap")
    assert out.read_bytes() == ref.read_bytes()


def test_the_table_round_trips_through_its_tsv_format(tmp_path: Path):
    t = table_for(COUNTS)
    p = tmp_path / "t.tsv"
    reap.write_table(t, p)
    assert p.read_text().splitlines()[0] == "layer\texpert\tcount\tsum_gate\tsum_saliency\tsum_norm"
    assert reap.load_table(p) == t
    p.write_text("layer\texpert\n0\t1\n")
    with pytest.raises(reap.ReapError, match="columns"):
        reap.load_table(p)


def test_a_table_from_an_importance_matrix_ranks_by_count_and_energy(tmp_path: Path):
    p = tmp_path / "im.gguf"
    make_imatrix(p, LAYERS, EXPERTS, COUNTS)
    t = reap.table_from_imatrix(p)
    assert set(t) == {0, 1, 2} and len(t[0]) == EXPERTS
    assert [t[0][e][0] for e in range(EXPERTS)] == COUNTS[0]
    assert reap.select(t, [0], EXPERTS, 4, "freq")[0] == [1, 3, 4, 6]
    assert reap.select(t, [0], EXPERTS, 4, "energy")[0] != []
    old = tmp_path / "old.gguf"
    ggufio.write_simple(old, [ggufio.kv_scalar("general.type", ggufio.T_STRING, "imatrix")], [])
    with pytest.raises(reap.ReapError, match="per-expert counts"):
        reap.table_from_imatrix(old)
