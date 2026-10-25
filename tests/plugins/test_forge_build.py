from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from forge_synth import make_model, make_split

from fle_studio_forge import build, ggufio, plan
from fle_studio_forge import recipe as recipes

LAYERS = 6


def mk(tmp_path: Path, name: str, seed: int, ttype: str) -> str:
    return str(make_model(tmp_path / f"{name}.gguf", seed=seed, layers=LAYERS, types=[(ttype,) * 3] * LAYERS))


def hybrid_recipe(files: dict[str, str], layers: dict[int, str], base: str = "B") -> dict:
    return recipes.normalize(
        {
            "name": "t",
            "mode": "hybrid",
            "sources": {sid: {"path": p, "role": "base" if sid == base else "donor"} for sid, p in files.items()},
            "hybrid": {"base": base, "layers": {str(k): v for k, v in layers.items()}},
        }
    )


def tensor_bytes(path: str, name: str) -> bytes:
    m = ggufio.open_model(path)
    p, off, n = m.location(name)
    with open(p, "rb") as f:
        f.seek(off)
        return f.read(n)


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    return {"B": mk(tmp_path, "base", 1, "Q8_0"), "K": mk(tmp_path, "k", 2, "Q5_0"), "L": mk(tmp_path, "l", 3, "Q4_0")}


def test_a_hybrid_takes_whole_layers_from_donors_and_the_rest_from_the_base(tmp_path: Path, files):
    layers = {0: "K", 1: "K", 3: "L", 5: "L"}
    r = hybrid_recipe(files, layers)
    out = tmp_path / "out" / "h.gguf"
    result = build.build_hybrid(r, out)
    assert out.exists() and not out.with_name("h.gguf.part").exists()
    assert result["size"] == out.stat().st_size
    assert result["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    base, hyb = ggufio.open_model(files["B"]), ggufio.open_model(str(out))
    assert hyb.order == base.order
    assert [k.encoded() for k in hyb.kv] == [k.encoded() for k in base.kv]  # metadata is carried over byte for byte
    for name in hyb.order:
        src = files["B"]
        m = plan.EXPERT_LAYER.match(name)
        if m and int(m.group(1)) in layers:
            src = files[layers[int(m.group(1))]]
        assert tensor_bytes(str(out), name) == tensor_bytes(src, name), name
        have, want = hyb.tensors[name], ggufio.open_model(src).tensors[name]
        assert (have.type, have.shape) == (want.type, want.shape), name
    assert result["origin"] == {"B": len(base.order) - 3 * 4, "K": 6, "L": 6}


def test_the_size_is_predicted_exactly_from_the_headers(tmp_path: Path, files):
    layers = {0: "K", 2: "L", 4: "K"}
    r = hybrid_recipe(files, layers)
    base = ggufio.open_model(files["B"])
    models = {sid: ggufio.open_model(p) for sid, p in files.items()}
    assigned = plan.assemble(base, models, {k: v for k, v in layers.items()}, "B")
    predicted = plan.predicted_file_size(base, models, assigned)
    out = tmp_path / "h.gguf"
    build.build_hybrid(r, out)
    assert out.stat().st_size == predicted


def test_a_hybrid_without_donor_layers_is_refused_by_the_step_not_the_builder(tmp_path: Path, files):
    # The builder itself copies whatever the recipe says; with no layers it reproduces the base bit for bit.
    r = hybrid_recipe(files, {})
    out = tmp_path / "copy.gguf"
    build.build_hybrid(r, out)
    assert out.read_bytes() == Path(files["B"]).read_bytes()


def test_a_split_donor_is_read_like_a_single_file(tmp_path: Path, files):
    split = make_split(tmp_path / "donor.gguf", seed=2, parts=3, layers=LAYERS)
    whole = tmp_path / "whole2.gguf"
    make_model(whole, seed=2, layers=LAYERS)
    out_a, out_b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    build.build_hybrid(hybrid_recipe({"B": files["B"], "K": str(split)}, {1: "K", 2: "K"}), out_a)
    build.build_hybrid(hybrid_recipe({"B": files["B"], "K": str(whole)}, {1: "K", 2: "K"}), out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_a_donor_with_another_shape_is_refused(tmp_path: Path, files):
    other = make_model(tmp_path / "wide.gguf", seed=7, layers=LAYERS, experts=4)
    r = hybrid_recipe({"B": files["B"], "K": str(other)}, {0: "K"})
    with pytest.raises(plan.PlanError, match="shape"):
        build.build_hybrid(r, tmp_path / "x.gguf")
    assert not (tmp_path / "x.gguf").exists()


def test_a_cancelled_build_resumes_and_gives_the_same_file(tmp_path: Path, files):
    layers = {0: "K", 1: "L", 2: "K", 3: "L"}
    r = hybrid_recipe(files, layers)
    reference = tmp_path / "ref.gguf"
    ref = build.build_hybrid(r, reference)

    out = tmp_path / "resumed.gguf"
    calls = {"n": 0}

    def cancel_after_some() -> bool:
        calls["n"] += 1
        return calls["n"] > 40  # a few writes are done, the rest is not

    with pytest.raises(ggufio.Cancelled):
        build.build_hybrid(r, out, cancelled=cancel_after_some)
    part = out.with_name("resumed.gguf.part")
    assert part.exists() and not out.exists()
    assert out.with_name("resumed.gguf.part.json").exists()
    assert 0 < part.stat().st_size < reference.stat().st_size

    lines: list[str] = []
    result = build.build_hybrid(r, out, log=lines.append)
    assert any(line.startswith("Resuming") for line in lines)
    assert out.read_bytes() == reference.read_bytes()
    assert result["sha256"] == ref["sha256"]
    assert not part.exists() and not out.with_name("resumed.gguf.part.json").exists()


def test_a_partial_file_of_another_recipe_is_not_continued(tmp_path: Path, files):
    out = tmp_path / "x.gguf"
    n = {"i": 0}

    def stop() -> bool:
        n["i"] += 1
        return n["i"] > 30

    with pytest.raises(ggufio.Cancelled):
        build.build_hybrid(hybrid_recipe(files, {0: "K"}), out, cancelled=stop)
    lines: list[str] = []
    build.build_hybrid(hybrid_recipe(files, {0: "L", 1: "L"}), out, log=lines.append)
    assert any("different recipe" in line for line in lines)
    expected = tmp_path / "e.gguf"
    build.build_hybrid(hybrid_recipe(files, {0: "L", 1: "L"}), expected)
    assert out.read_bytes() == expected.read_bytes()


def test_a_truncated_source_is_reported(tmp_path: Path, files):
    data = Path(files["K"]).read_bytes()
    Path(files["K"]).write_bytes(data[: len(data) - 5000])
    with pytest.raises(ggufio.GGUFError):
        build.build_hybrid(hybrid_recipe(files, {5: "K"}), tmp_path / "t.gguf")


def test_verify_file_finds_a_donor_layer_and_a_flipped_byte(tmp_path: Path, files):
    out = tmp_path / "h.gguf"
    build.build_hybrid(hybrid_recipe(files, {0: "K"}), out)
    model = ggufio.open_model(files["B"])
    all_base = [
        build.OutTensor(n, model.tensors[n].shape, model.tensors[n].type, model.tensors[n].nbytes, [model.location(n)])
        for n in model.order
    ]
    # Checked against "everything from the base", only the three tensors of the donor layer may differ.
    bad = {p.split(": ", 1)[1] for p in build.verify_file(out, model.kv, all_base, 32)}
    assert bad == {f"blk.0.ffn_{k}_exps.weight" for k in ("gate", "up", "down")}

    copy = tmp_path / "copy.gguf"
    build.build_hybrid(hybrid_recipe(files, {}), copy)
    assert build.verify_file(copy, model.kv, all_base, 32) == []
    g = ggufio.read_gguf(copy)
    raw = bytearray(copy.read_bytes())
    raw[g.data_start + g.tensors[0].offset] ^= 0xFF
    copy.write_bytes(bytes(raw))
    assert build.verify_file(copy, model.kv, all_base, 32) == [f"first bytes: {g.tensors[0].name}"]
