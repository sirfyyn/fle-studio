from __future__ import annotations

from pathlib import Path

import pytest
from forge_synth import HIDDEN, INTER, make_model, make_split

from fle_studio import gguf
from fle_studio_forge import ggufio


def test_type_sizes_follow_the_block_layouts():
    q4 = ggufio.type_id("q4_0")
    assert ggufio.row_bytes(q4, 64) == 36
    assert ggufio.row_bytes(q4, 33) is None  # not a whole number of blocks
    assert ggufio.tensor_nbytes(ggufio.type_id("IQ3_S"), (256, 4)) == 4 * 110
    assert ggufio.tensor_nbytes(ggufio.type_id("F32"), (5, 3)) == 60
    assert ggufio.bits_per_weight(ggufio.type_id("Q8_0")) == pytest.approx(8.5)
    assert ggufio.bits_per_weight(ggufio.type_id("IQ4_NL")) == pytest.approx(4.5)
    assert ggufio.type_name(9999) == "TYPE_9999"
    assert ggufio.type_id("nonsense") is None


def test_the_reader_keeps_the_raw_metadata_and_the_tensor_table(tmp_path: Path):
    path = make_model(tmp_path / "m.gguf", seed=1, layers=2)
    g = ggufio.read_gguf(path)
    assert g.version == 3 and g.alignment == 32
    keys = [k.key for k in g.kv]
    assert keys[0] == "general.architecture" and "tokenizer.ggml.tokens" in keys
    tokens = next(k for k in g.kv if k.key == "tokenizer.ggml.tokens")
    assert tokens.value is None and tokens.vtype == ggufio.T_ARRAY and len(tokens.raw) > 16
    names = [t.name for t in g.tensors]
    assert "blk.1.ffn_down_exps.weight" in names
    down = next(t for t in g.tensors if t.name == "blk.0.ffn_down_exps.weight")
    assert down.shape == (INTER, HIDDEN, 8) and down.nbytes == 18 * HIDDEN * 8  # Q4_0: 18 bytes per row of 32
    assert g.data_start % 32 == 0 and g.data_start + g.tensors[-1].offset + g.tensors[-1].nbytes <= g.file_size


def test_sizes_agree_with_the_existing_header_reader(tmp_path: Path):
    path = make_model(tmp_path / "m.gguf", seed=2)
    mine = {t.name: t.nbytes for t in ggufio.read_gguf(path).tensors}
    theirs = {t.name: t.size for t in gguf.read_header(path).tensors}
    assert set(mine) == set(theirs)
    for name in mine:  # the existing reader includes the padding up to the next tensor
        assert 0 <= theirs[name] - mine[name] < 32


def test_header_is_deterministic_and_round_trips(tmp_path: Path):
    path = make_model(tmp_path / "m.gguf", seed=3)
    g = ggufio.read_gguf(path)
    infos = [(t.name, t.shape, t.type, t.nbytes) for t in g.tensors]
    header, offsets = ggufio.build_header(g.kv, infos, g.alignment)
    assert header == path.read_bytes()[: len(header)]
    assert offsets == [t.offset for t in g.tensors]
    assert len(header) == g.data_start


def test_with_value_replaces_one_scalar_and_nothing_else(tmp_path: Path):
    g = ggufio.read_gguf(make_model(tmp_path / "m.gguf", seed=4))
    changed = ggufio.with_value(g.kv, "testmoe.expert_count", 5)
    before = {k.key: k.raw for k in g.kv}
    after = {k.key: k.raw for k in changed}
    assert [k for k in before if before[k] != after[k]] == ["testmoe.expert_count"]
    assert next(k for k in changed if k.key == "testmoe.expert_count").value == 5
    with pytest.raises(gguf.GGUFError):
        ggufio.with_value(g.kv, "no.such.key", 1)
    with pytest.raises(gguf.GGUFError):
        ggufio.with_value(g.kv, "general.name", 1)


def test_a_split_model_is_one_model_without_split_keys(tmp_path: Path):
    first = make_split(tmp_path / "s.gguf", seed=5, parts=3)
    m = ggufio.open_model(first)
    assert len(m.files) == 3 and m.paths[0] == str(first)
    assert not any(k.key.startswith("split.") for k in m.kv)
    whole = ggufio.open_model(make_model(tmp_path / "w.gguf", seed=5))
    assert m.order == whole.order
    for name in m.order[:6]:
        a_path, a_off, a_n = m.location(name)
        b_path, b_off, b_n = whole.location(name)
        with open(a_path, "rb") as fa, open(b_path, "rb") as fb:
            fa.seek(a_off)
            fb.seek(b_off)
            assert fa.read(a_n) == fb.read(b_n)


def test_a_missing_part_is_named(tmp_path: Path):
    first = make_split(tmp_path / "s.gguf", seed=6, parts=2)
    first.with_name(first.name.replace("00001", "00002")).unlink()
    with pytest.raises(gguf.GGUFError, match="00002"):
        ggufio.open_model(first)


def test_a_file_that_is_not_gguf_is_refused(tmp_path: Path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"not a model" * 10)
    with pytest.raises(gguf.GGUFError):
        ggufio.read_gguf(p)
