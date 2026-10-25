from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from forge_synth import make_model

from fle_studio_forge import ggufio, plan

LAYERS = 6
TYPES = {
    "L": [("Q4_0", "Q4_0", "Q4_0")] * LAYERS,  # low
    "M": [("Q5_0", "Q5_0", "Q5_0")] * LAYERS,  # middle
    "H": [("Q8_0", "Q8_0", "Q8_0")] * LAYERS,  # high
}


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, plan.Source]:
    out = {}
    for i, (sid, types) in enumerate(TYPES.items()):
        out[sid] = plan.Source.load(sid, str(make_model(tmp_path / f"{sid}.gguf", seed=i, layers=LAYERS, types=types)))
    return out


def header_bytes(src: plan.Source) -> int:
    m = src.model
    infos = [(n, t.shape, t.type, t.nbytes) for n, t in m.tensors.items()]
    return len(ggufio.build_header(m.kv, infos, m.alignment)[0])


def brute(sources: dict[str, plan.Source], base: str, budget: int, errors=None) -> float | None:
    err = plan.layer_errors(sources, errors)
    b = sources[base]
    rest = b.rest_bytes + header_bytes(b)
    best = None
    for combo in itertools.product(sources, repeat=len(b.layers)):
        size = rest + sum(sources[s].layer_bytes[layer] for s, layer in zip(combo, b.layers, strict=True))
        if size <= budget:
            e = sum(err[s][layer] for s, layer in zip(combo, b.layers, strict=True))
            best = e if best is None else min(best, e)
    return best


def test_describe_reports_types_and_sizes_per_layer(sources):
    d = sources["L"].describe()
    assert d["layers"] == LAYERS and d["experts"] == 8 and d["experts_used"] == 2
    assert d["expert_layers"] == LAYERS
    assert d["types"] == [{"types": "down Q4_0 / gate Q4_0 / up Q4_0", "layers": list(range(LAYERS))}]
    assert sources["H"].describe()["expert_bytes"] > d["expert_bytes"]
    assert d["bpw"] == pytest.approx(4.5)


@pytest.mark.parametrize("fraction", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_the_choice_is_optimal_for_the_budget(sources, fraction):
    base = "H"
    hb = header_bytes(sources[base])
    lo = sum(min(s.layer_bytes[x] for s in sources.values()) for x in sources[base].layers)
    hi = sum(max(s.layer_bytes[x] for s in sources.values()) for x in sources[base].layers)
    rest = sources[base].rest_bytes + hb
    budget = rest + lo + int((hi - lo) * fraction)
    sel = plan.choose_layers(sources, base, budget, header_bytes=hb)
    assert sel.feasible and sel.bytes <= budget
    assert set(sel.layers) == set(range(LAYERS))
    best_here = brute(sources, base, budget)
    best_less = brute(sources, base, budget - 200)  # the knapsack rounds sizes up, by far less than 200 bytes
    assert sel.error >= best_here - 1e-12
    assert sel.error <= best_less + 1e-12


def test_a_budget_above_everything_picks_the_best_of_every_layer(sources):
    hb = header_bytes(sources["H"])
    sel = plan.choose_layers(sources, "H", 10**9, header_bytes=hb)
    assert set(sel.layers.values()) == {"H"}
    assert sel.bytes == sel.max_bytes


def test_a_budget_below_the_smallest_result_is_reported_not_guessed(sources):
    sel = plan.choose_layers(sources, "H", 1000, header_bytes=header_bytes(sources["H"]))
    assert not sel.feasible and "below the smallest" in sel.note
    assert set(sel.layers.values()) == {"L"}


def test_measured_errors_beat_the_stand_in(sources):
    # Layer 2 is very sensitive: give it the best source even though the stand-in would spread the bits evenly.
    errors = {
        sid: {str(x): (0.001 if x != 2 else 5.0) * (4.0 ** -plan_bpw(s, x)) for x in s.layers}
        for sid, s in sources.items()
    }
    hb = header_bytes(sources["H"])
    lo = sum(min(s.layer_bytes[x] for s in sources.values()) for x in sources["H"].layers)
    budget = sources["H"].rest_bytes + hb + lo + (sources["H"].layer_bytes[2] - sources["L"].layer_bytes[2]) + 50
    sel = plan.choose_layers(sources, "H", budget, errors, hb)
    assert sel.layers[2] == "H"
    assert sel.error >= brute(sources, "H", budget, errors) - 1e-12


def plan_bpw(src: plan.Source, layer: int) -> float:
    return src.bpw(layer)


def test_missing_error_entries_are_named(sources):
    full = {str(x): 1.0 for x in range(LAYERS)}
    with pytest.raises(plan.PlanError, match="no entry for source M"):
        plan.choose_layers(sources, "H", 10**9, {"L": full, "H": full})
    with pytest.raises(plan.PlanError, match="lacks layer"):
        plan.choose_layers(sources, "H", 10**9, {"L": full, "M": full, "H": {"0": 1.0}})


def test_models_that_are_not_quants_of_each_other_are_refused(tmp_path: Path, sources):
    other = plan.Source.load("X", str(make_model(tmp_path / "x.gguf", seed=9, layers=LAYERS, experts=4)))
    with pytest.raises(plan.PlanError, match="differ in name or shape"):
        plan.choose_layers({**sources, "X": other}, "H", 10**9)
    fewer = plan.Source.load("Y", str(make_model(tmp_path / "y.gguf", seed=9, layers=3)))
    with pytest.raises(plan.PlanError, match="expert layers"):
        plan.choose_layers({**sources, "Y": fewer}, "H", 10**9)


def test_assemble_rejects_a_donor_without_the_tensor(tmp_path: Path, sources):
    small = ggufio.open_model(make_model(tmp_path / "s.gguf", seed=1, layers=3))
    base = sources["H"].model
    with pytest.raises(plan.PlanError, match="no tensor"):
        plan.assemble(base, {"S": small}, {5: "S"}, "H")


def test_the_memory_preview_separates_experts_from_the_rest(sources):
    base = sources["H"].model
    models = {"H": base}
    assigned = {n: "H" for n in base.order}
    big = plan.memory_preview(base, models, assigned, ram_bytes=64 * plan.GIB, vram_bytes=24 * plan.GIB)
    assert big["verdict"] == "all_gpu"
    assert big["experts"] > 0 and big["dense"] > 0 and big["inputs"] > 0
    assert big["experts"] + big["dense"] + big["inputs"] == sum(t.nbytes for t in base.tensors.values())
    cpu = plan.memory_preview(base, models, assigned, ram_bytes=64 * plan.GIB, vram_bytes=0)
    assert cpu["verdict"] == "cpu_only"
    tiny = plan.memory_preview(base, models, assigned, ram_bytes=plan.GIB, vram_bytes=0)
    assert tiny["verdict"] == "too_big"
    # Just enough VRAM for everything except the experts: they stay in RAM.
    need = big["experts_on_cpu"]["vram"]
    squeezed = plan.memory_preview(base, models, assigned, 64 * plan.GIB, need + 10)
    assert squeezed["verdict"] == "experts_cpu"
    assert squeezed["experts_on_cpu"]["gpu_cache"] == 10
    short = plan.memory_preview(base, models, assigned, 64 * plan.GIB, need - 10)
    assert short["verdict"] == "cpu_only"  # does not fit the GPU at all: everything runs from RAM


def test_the_quantize_estimate_follows_the_rules(sources):
    m = sources["H"].model
    plain = plan.estimate_quantize(m, "q4_0", [])
    ruled = plan.estimate_quantize(m, "q4_0", [{"pattern": "ffn_.*_exps", "type": "q8_0"}])
    assert ruled["estimate_bytes"] > plain["estimate_bytes"]
    assert plain["estimate"] is True and plain["tensors_kept"] > 0  # vectors keep their type
    same = plan.estimate_quantize(m, "q8_0", [])
    assert same["estimate_bytes"] == pytest.approx(m.total_bytes, rel=0.01)


def test_a_named_mixture_is_estimated_by_its_main_type(sources):
    m = sources["H"].model
    mix = plan.estimate_quantize(m, "q3_k_m", [])
    assert mix["estimate_bytes"] == plan.estimate_quantize(m, "q3_k", [])["estimate_bytes"]
