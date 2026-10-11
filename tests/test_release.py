"""The version is named in several places; this keeps them from drifting apart before a release.

While a version is in development (`0.2.0.dev0`) the installers and the README still name the last release, the one
that exists on GitHub; the changelog then starts with an "Unreleased" section above it.
"""

from __future__ import annotations

import re
from pathlib import Path

from fle_studio import __version__

ROOT = Path(__file__).resolve().parent.parent
DEV = ".dev" in __version__


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def headings() -> list[str]:
    return re.findall(r"^## \[([^\]]+)\]", read("CHANGELOG.md"), re.M)


def released() -> str:
    """The newest released version: the first numbered heading of the changelog."""
    numbered = [h for h in headings() if re.fullmatch(r"\d+\.\d+\.\d+", h)]
    assert numbered, "CHANGELOG.md has no '## [x.y.z]' heading"
    return numbered[0]


def tuple_of(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.match(r"\d+\.\d+\.\d+", version).group(0).split("."))  # type: ignore[union-attr]


def test_version_looks_like_a_release_or_a_development_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", __version__)


def test_installers_default_to_the_last_release():
    v = released()
    assert f"FLE_STUDIO_REF:-v{v}}}" in read("install.sh")
    assert f"else {{ 'v{v}' }}" in read("install.ps1")


def test_installers_say_the_same_default_in_their_help():
    v = released()
    assert f"(default: v{v})" in read("install.sh")
    assert f"else v{v}." in read("install.ps1")


def test_readme_names_the_last_release():
    assert f"(v{released()} at the time of writing)" in read("README.md")


def test_changelog_matches_the_version():
    found = headings()
    if DEV:
        assert found[0] == "Unreleased", "a development version needs an 'Unreleased' section at the top"
        assert tuple_of(__version__) > tuple_of(released())
    else:
        assert found[0] == __version__
