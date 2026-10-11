"""Runs the browser-side checks under Node. Skipped when Node is not installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "js" / "check_request.mjs"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_request_and_tree_code() -> None:
    node = shutil.which("node")
    assert node
    done = subprocess.run([node, "--test", str(SCRIPT)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
