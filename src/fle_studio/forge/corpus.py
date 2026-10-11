"""Calibration text for the importance matrix, the router statistics and the KL-divergence check.

The quality of a quant made with an importance matrix depends on the text it was calibrated on: the matrix says which
weights matter for *that* text. A mixture-of-experts model needs a lot of it (a short text never routes to many of
the experts). Forge therefore mixes what you give it: text files and folders of text, cut into units and drawn at
random (with a fixed seed, so the same inputs give the same file) up to a character budget, with a per-source report
so that no single source silently dominates.

A small mixed text ships with Studio (`bundled()`, about 1 MB: English and German prose, code, dialogue, numbers; every
file with its origin and license in corpus/SOURCES.md). It is a sound default for a first quant, not a substitute for
text from your own use. For a trial `stdlib_sample()` collects documentation and code from the Python standard
library of the interpreter that runs Studio.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".js",
    ".ts",
    ".json",
    ".sh",
    ".html",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".tex",
    ".csv",
}
UNIT = re.compile(r"\n\s*\n")


class CorpusError(ValueError):
    pass


def _files(inputs: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_file():
            out[str(p)] = [p]
        elif p.is_dir():
            found = sorted(
                f
                for f in p.rglob("*")
                if f.is_file()
                and f.suffix.lower() in TEXT_SUFFIXES
                and not any(part.startswith(".") for part in f.relative_to(p).parts)
            )
            out[str(p)] = found
        else:
            raise CorpusError(f"{item} is neither a file nor a folder")
    return out


def build(
    inputs: list[str],
    out: str | os.PathLike[str],
    chars: int = 2_000_000,
    seed: int = 1,
    max_unit: int = 4000,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Mix the inputs into one text file. Every input gets an equal share of the budget; what an input cannot fill
    goes to the others. Returns the per-input report."""
    if chars < 1000:
        raise CorpusError("a corpus of fewer than 1000 characters is of no use")
    groups = _files(inputs)
    units: dict[str, list[str]] = {}
    for key, files in groups.items():
        found: list[str] = []
        for f in files:
            if cancelled():
                raise CorpusError("cancelled")
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            found += [u.strip()[:max_unit] for u in UNIT.split(text) if len(u.strip()) >= 80]
        if not found:
            raise CorpusError(f"{key} holds no usable text (looked at {len(files)} file(s))")
        units[key] = found
    rng = random.Random(seed)
    for u in units.values():
        rng.shuffle(u)
    budget = {k: chars / len(units) for k in units}
    chosen: dict[str, list[str]] = {k: [] for k in units}
    for _ in range(4):  # whatever a small input cannot fill is handed to the others
        spare = 0.0
        for k, u in units.items():
            while budget[k] > 0 and u:
                unit = u.pop()
                chosen[k].append(unit)
                budget[k] -= len(unit) + 2
            if budget[k] > 0:
                spare += budget[k]
                budget[k] = 0
        open_ = [k for k in units if units[k]]
        if spare <= 0 or not open_:
            break
        for k in open_:
            budget[k] += spare / len(open_)
    everything = [x for k in units for x in chosen[k]]
    rng.shuffle(everything)
    text = "\n\n".join(everything)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    total = max(1, sum(len(x) for k in chosen for x in chosen[k]))
    report = [
        {
            "input": k,
            "units": len(chosen[k]),
            "chars": sum(len(x) for x in chosen[k]),
            "share": sum(len(x) for x in chosen[k]) / total,
        }
        for k in chosen
    ]
    return {
        "path": str(out),
        "chars": len(text),
        "words": len(text.split()),
        "sources": report,
        "largest_share": max(r["share"] for r in report),
    }


BUNDLED_DIR = Path(__file__).parent / "corpus"
HELD_OUT_PERCENT = 15


def _held_out(unit: str) -> bool:
    return int(hashlib.sha256(unit.encode("utf-8")).hexdigest(), 16) % 100 < HELD_OUT_PERCENT


def bundled(
    out: str | os.PathLike[str], part: str = "calibration", seed: int = 1, max_unit: int = 4000
) -> dict[str, Any]:
    """The text that ships with Studio (see corpus/SOURCES.md for origin and license of every file).

    "calibration" is about 85 % of its paragraphs and "evaluation" the other 15 %, chosen by a hash of the paragraph,
    so the two never share a paragraph: use the first for the importance matrix and the router statistics, the second
    for the comparison with the reference model."""
    if part not in ("calibration", "evaluation"):
        raise CorpusError("part must be calibration or evaluation")
    files = sorted(BUNDLED_DIR.glob("*.txt"))
    if not files:
        raise CorpusError("this installation has no bundled corpus")
    units: list[str] = []
    per_file: dict[str, int] = {}
    for f in files:
        before = len(units)
        for u in UNIT.split(f.read_text(encoding="utf-8", errors="ignore")):
            u = u.strip()[:max_unit]
            if len(u) >= 80 and _held_out(u) == (part == "evaluation"):
                units.append(u)
        per_file[f.name] = sum(len(x) for x in units[before:])
    random.Random(seed).shuffle(units)
    text = "\n\n".join(units)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    total = max(1, sum(per_file.values()))
    report = [{"input": k, "chars": v, "share": v / total} for k, v in per_file.items()]
    return {
        "path": str(out),
        "chars": len(text),
        "words": len(text.split()),
        "sources": report,
        "largest_share": max(r["share"] for r in report),
        "part": part,
    }


def stdlib_sample(out: str | os.PathLike[str], chars: int = 1_000_000, seed: int = 1) -> dict[str, Any]:
    lib = sysconfig.get_paths().get("stdlib")
    if not lib or not Path(lib).is_dir():
        raise CorpusError("this Python has no standard library folder to sample from")
    return build([lib], out, chars=chars, seed=seed)
