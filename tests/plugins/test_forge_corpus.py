"""The calibration text that ships with Studio: size, split, and a documented origin for every file."""

from __future__ import annotations

import hashlib
import importlib.resources as resources
import re
from pathlib import Path

import pytest

from fle_studio_forge import corpus


def test_the_corpus_is_about_a_megabyte_and_mixed(tmp_path: Path):
    made = corpus.bundled(tmp_path / "c.txt")
    assert 500_000 < made["chars"] < 2_000_000
    names = {s["input"] for s in made["sources"]}
    assert {"tables.txt", "code.txt", "dialog.txt"} <= names
    assert any(n.startswith("prose-en-") for n in names) and any(n.startswith("prose-de-") for n in names)
    assert made["largest_share"] < 0.3  # no single file dominates
    text = (tmp_path / "c.txt").read_text(encoding="utf-8")
    assert "Alice" in text and "Effi" in text and "def " in text and "|" in text


def test_calibration_and_evaluation_share_no_paragraph(tmp_path: Path):
    cal = corpus.bundled(tmp_path / "a.txt", "calibration")
    ev = corpus.bundled(tmp_path / "b.txt", "evaluation")
    a = set((tmp_path / "a.txt").read_text(encoding="utf-8").split("\n\n"))
    b = set((tmp_path / "b.txt").read_text(encoding="utf-8").split("\n\n"))
    assert a and b and not a & b
    assert 0.08 < ev["chars"] / (ev["chars"] + cal["chars"]) < 0.25
    with pytest.raises(corpus.CorpusError):
        corpus.bundled(tmp_path / "c.txt", "training")


def test_the_corpus_is_the_same_every_time(tmp_path: Path):
    corpus.bundled(tmp_path / "a.txt")
    corpus.bundled(tmp_path / "b.txt")
    assert (tmp_path / "a.txt").read_bytes() == (tmp_path / "b.txt").read_bytes()


def test_every_file_has_origin_license_and_a_matching_checksum():
    folder = resources.files("fle_studio_forge").joinpath("corpus")
    sources = folder.joinpath("SOURCES.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| (\d+) \| `([0-9a-f]{64})` \| (.+) \|$", sources, re.M)
    listed = {name for name, *_ in rows}
    on_disk = {p.name for p in Path(str(folder)).glob("*.txt")}
    assert listed == on_disk and len(rows) >= 8
    for name, size, digest, note in rows:
        data = Path(str(folder.joinpath(name))).read_bytes()
        assert len(data) == int(size), name
        assert hashlib.sha256(data).hexdigest() == digest, f"{name} changed; run scripts/make_corpus.py"
        assert re.search(r"public domain|CC0|MIT License", note, re.I), f"{name} names no license"
        assert re.search(r"Project Gutenberg #\d+|pytest|FLE Studio|seeded|Written for", note), (
            f"{name} names no origin"
        )


def test_no_gutenberg_license_text_is_left_in_the_prose():
    folder = Path(str(resources.files("fle_studio_forge").joinpath("corpus")))
    for p in folder.glob("prose-*.txt"):
        text = p.read_text(encoding="utf-8")
        assert "Project Gutenberg" not in text and "www.gutenberg.org" not in text, p.name
