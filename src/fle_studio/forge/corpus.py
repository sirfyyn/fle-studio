"""Calibration text for the importance matrix, the router statistics and the KL-divergence check.

The quality of a quant made with an importance matrix depends on the text it was calibrated on: the matrix says which
weights matter for *that* text. A mixture-of-experts model needs a lot of it (a short text never routes to many of
the experts). Forge therefore mixes what you give it: text files and folders of text, cut into units and drawn at
random (with a fixed seed, so the same inputs give the same file) up to a character budget, with a per-source report
so that no single source silently dominates.

No corpus is bundled. For a first trial `stdlib_sample()` collects English documentation and code from the Python
standard library of the interpreter that runs Studio; it is a convenience, not a recommendation.
"""

from __future__ import annotations

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


def stdlib_sample(out: str | os.PathLike[str], chars: int = 1_000_000, seed: int = 1) -> dict[str, Any]:
    lib = sysconfig.get_paths().get("stdlib")
    if not lib or not Path(lib).is_dir():
        raise CorpusError("this Python has no standard library folder to sample from")
    return build([lib], out, chars=chars, seed=seed)
