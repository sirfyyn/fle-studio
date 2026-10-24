"""Where FLE Studio keeps its files on each operating system."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Evaluated once at import. Plain booleans (not `sys.platform` comparisons inline) so that every branch is
# reachable for type checkers on any host and the three platforms read the same way everywhere.
IS_MAC: bool = sys.platform == "darwin"
IS_WIN: bool = os.name == "nt"
IS_LINUX: bool = sys.platform.startswith("linux")


def data_dir() -> Path:
    """The per-user data folder: settings, chats, engines and (by default) models.

    FLE_STUDIO_HOME overrides it, which the tests and portable installs use.
    """
    override = os.environ.get("FLE_STUDIO_HOME")
    if override:
        return Path(override).expanduser()
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / "FLE Studio"
    if IS_WIN:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "FLE Studio"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "fle-studio"


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def exe(name: str) -> str:
    """File name of an executable on this platform."""
    return f"{name}.exe" if IS_WIN else name
