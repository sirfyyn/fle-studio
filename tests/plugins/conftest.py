"""The tests of the plugins that ship with FLE Studio. They need those plugins: with FLE_STUDIO_NO_BUNDLED=1 (the
core alone, see tests/test_core_alone.py and the CI) they are not collected."""

from __future__ import annotations

import os

collect_ignore_glob = (
    ["test_*.py"] if os.environ.get("FLE_STUDIO_NO_BUNDLED", "").strip().lower() in ("1", "true", "yes", "on") else []
)
