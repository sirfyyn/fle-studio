"""Builds the calibration corpus that ships with Forge (src/fle_studio/forge/corpus/).

    python scripts/make_corpus.py

Needs internet (Project Gutenberg) and an installed pytest (MIT code samples). The files are committed, so nobody has
to run this; it exists so that the corpus is reproducible and every part has a documented origin. It writes the
component files and SOURCES.md (origin, license, retrieval date and SHA-256 of every file).
"""

from __future__ import annotations

import datetime
import hashlib
import random
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "fle_studio" / "forge" / "corpus"
PG = "https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt"

# (id, title, author, language, characters to keep)
BOOKS = [
    (11, "Alice's Adventures in Wonderland", "Lewis Carroll", "en", 110_000),
    (1342, "Pride and Prejudice", "Jane Austen", "en", 130_000),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "en", 130_000),
    (22367, "Die Verwandlung", "Franz Kafka", "de", 90_000),
    (21000, "Faust: Der Tragoedie erster Teil", "Johann Wolfgang von Goethe", "de", 70_000),
    (5323, "Effi Briest", "Theodor Fontane", "de", 110_000),
]
START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8-sig", "replace")


def strip_gutenberg(text: str) -> str:
    """Only the work itself: the Project Gutenberg license head and foot are removed."""
    m = START.search(text)
    if m:
        text = text[m.end() :]
    m = END.search(text)
    if m:
        text = text[: m.start()]
    return re.sub(r"\r\n?", "\n", text).strip()


def cut(text: str, chars: int) -> str:
    """The first `chars` characters, ended at a paragraph boundary, after skipping front matter."""
    start = min(len(text) // 10, 6000)
    start = (text.find("\n\n", start) + 2) if text.find("\n\n", start) >= 0 else start
    body = text[start:]
    end = body.rfind("\n\n", 0, chars)
    return body[: end if end > 0 else chars].strip() + "\n"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prose() -> list[tuple[Path, str]]:
    notes = []
    today = datetime.date.today().isoformat()
    for n, title, author, lang, chars in BOOKS:
        text = cut(strip_gutenberg(fetch(PG.format(n=n))), chars)
        path = OUT / f"prose-{lang}-{n}.txt"
        path.write_text(text, encoding="utf-8", newline="\n")
        notes.append(
            (
                path,
                f"{title}, {author}. Project Gutenberg #{n} ({PG.format(n=n)}), retrieved {today}. "
                f"Public domain (author died before 1930). License head and foot removed.",
            )
        )
    return notes


def code() -> list[tuple[Path, str]]:
    import _pytest

    root = Path(_pytest.__file__).parent
    from importlib.metadata import version

    ver = version("pytest")
    parts = []
    for name in ("python_api.py", "fixtures.py", "monkeypatch.py", "tmpdir.py", "capture.py", "recwarn.py"):
        parts.append(f"# ---- _pytest/{name} (pytest {ver})\n" + (root / name).read_text(encoding="utf-8"))
    own = Path(__file__).resolve().parent.parent / "src" / "fle_studio"
    for name in ("web/js/ui.js", "web/js/api.js", "config.py", "paths.py"):
        parts.append(f"// ---- fle_studio/{name}\n" + (own / name).read_text(encoding="utf-8"))
    path = OUT / "code.txt"
    path.write_text("\n\n".join(parts).replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    return [
        (
            path,
            f"Python from pytest {ver} (MIT License, Copyright (c) 2004 Holger Krekel and others), "
            "taken from the installed package; JavaScript and Python from FLE Studio itself (MIT License, "
            "Copyright (c) 2026 Max Bitzer (sirfyyn)).",
        )
    ]


def tables() -> list[tuple[Path, str]]:
    rng = random.Random(20261011)
    lines = ["# Measurements and tables", ""]
    cities = [
        "Berlin",
        "Hamburg",
        "Munich",
        "Cologne",
        "Leipzig",
        "Dresden",
        "Essen",
        "Bremen",
        "Hanover",
        "Bonn",
        "Lyon",
        "Porto",
        "Graz",
        "Basel",
        "Utrecht",
        "Ghent",
        "Turin",
        "Seville",
        "Gdansk",
        "Brno",
    ]
    for block in range(40):
        lines.append(f"## Table {block + 1}: monthly figures")
        lines.append("| city | month | temperature (C) | rain (mm) | sunshine (h) | visitors |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for _ in range(14):
            lines.append(
                f"| {rng.choice(cities)} | {rng.randint(1, 12):02d} | {rng.uniform(-6, 31):.1f} | "
                f"{rng.uniform(0, 160):.0f} | {rng.uniform(30, 290):.0f} | {rng.randint(900, 480000)} |"
            )
        lines.append("")
    lines.append("## Sensor log")
    for i in range(500):
        lines.append(
            f"2026-03-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d}Z sensor-{rng.randint(1, 40):02d} value={rng.gauss(20, 6):.3f} "
            f"unit=C status={rng.choice(['ok', 'ok', 'ok', 'warn', 'retry'])} seq={i}"
        )
    lines.append("")
    lines.append("## Arithmetic")
    for _ in range(300):
        a, b = rng.randint(2, 9999), rng.randint(2, 999)
        lines.append(f"{a} + {b} = {a + b}; {a} - {b} = {a - b}; {a} * {b} = {a * b}; {a} / {b} = {a / b:.4f}")
    lines.append("")
    path = OUT / "tables.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return [
        (
            path,
            "Generated by scripts/make_corpus.py from a seeded random number generator: invented figures. "
            "CC0 (no rights reserved).",
        )
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notes = prose() + code() + tables()
    handwritten = OUT / "dialog.txt"
    if handwritten.exists():
        notes.append(
            (
                handwritten,
                "Written for FLE Studio: questions and answers, English and German. CC0 (no rights "
                "reserved). Kept by hand; this script does not touch it.",
            )
        )
    rows = ["| file | size | SHA-256 | origin and license |", "| --- | ---: | --- | --- |"]
    total = 0
    for path, note in sorted(notes, key=lambda x: x[0].name):
        total += path.stat().st_size
        rows.append(f"| `{path.name}` | {path.stat().st_size} | `{sha(path)}` | {note} |")
    header = (
        "# Sources of the calibration corpus\n\n"
        "A mixed text for the importance matrix and the router statistics: English and German prose, code, dialogue "
        "and numbers. Every part may be redistributed under the license named here. The texts of Project Gutenberg "
        "are public domain in the United States and are used here without the Project Gutenberg license head. "
        f"Total {total} bytes. Built by `scripts/make_corpus.py`.\n\n"
        "The MIT notice of pytest, which applies to the code samples taken from it:\n\n"
        "> Copyright (c) 2004 Holger Krekel and others. Permission is hereby granted, free of charge, to any person "
        'obtaining a copy of this software and associated documentation files (the "Software"), to deal in the '
        "Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, "
        "distribute, sublicense, and/or sell copies of the Software, subject to the inclusion of this notice. The "
        'Software is provided "as is", without warranty of any kind.\n\n'
    )
    (OUT / "SOURCES.md").write_text(header + "\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(notes)} files, {total} bytes")


if __name__ == "__main__":
    main()
