"""The tests of the plugins that ship with FLE Studio. They need those plugins: with FLE_STUDIO_NO_BUNDLED=1 (the
core alone, see tests/test_core_alone.py and the CI) they are not collected."""

from __future__ import annotations

import os

collect_ignore_glob = (
    ["test_*.py"] if os.environ.get("FLE_STUDIO_NO_BUNDLED", "").strip().lower() in ("1", "true", "yes", "on") else []
)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_update_loops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every Studio of a test run starts the daily update check of the updater plugin. Those threads outlive their test
    and would ask the stand-in GitHub of a later test. The loop itself is not under test here."""
    try:
        from fle_studio_updater import updates
    except ImportError:
        return
    monkeypatch.setattr(updates, "start_background", lambda studio: lambda: None)
