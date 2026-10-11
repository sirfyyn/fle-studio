"""The version is named in several places; this keeps them from drifting apart before a release."""

from __future__ import annotations

import re
from pathlib import Path

from fle_studio import __version__

ROOT = Path(__file__).resolve().parent.parent


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_version_looks_like_a_release():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_installers_default_to_this_release():
    assert f'FLE_STUDIO_REF:-v{__version__}}}' in read("install.sh")
    assert f"else {{ 'v{__version__}' }}" in read("install.ps1")


def test_installers_say_the_same_default_in_their_help():
    assert f"(default: v{__version__})" in read("install.sh")
    assert f"else v{__version__}." in read("install.ps1")


def test_readme_names_this_release():
    assert f"(v{__version__} at the time of writing)" in read("README.md")


def test_changelog_has_this_release_first():
    versions = re.findall(r"^## \[(\d+\.\d+\.\d+)\]", read("CHANGELOG.md"), re.M)
    assert versions, "CHANGELOG.md has no '## [x.y.z]' heading"
    assert versions[0] == __version__
