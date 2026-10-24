from __future__ import annotations

import shutil
import sys
from pathlib import Path

from fle_studio.forge import tools

FAKES = Path(__file__).parent / "fake_tools"


def test_fake_tools_are_found_probed_and_report_their_features():
    found = tools.find([("settings", FAKES)], use_path=False)
    assert set(found) == set(tools.TOOL_NAMES)
    q = found["llama-quantize"]
    assert q.ok and q.version == "9999 (abc1234)" and q.source == "settings"
    assert q.features["tensor_type"] and q.features["imatrix"] and q.features["allow_requantize"]
    assert found["llama-perplexity"].features["kl_divergence"] and found["llama-fle-reap"].features["cmoe"]
    assert tools.command_for(q.path) == [sys.executable, q.path]
    assert tools.command_for("/usr/bin/llama-quantize") == ["/usr/bin/llama-quantize"]


def test_the_search_order_is_settings_then_engines(tmp_path: Path):
    other = tmp_path / "engine"
    other.mkdir()
    shutil.copy(FAKES / "llama-quantize.py", other / "llama-quantize.py")
    dirs = tools.search_dirs(str(tmp_path / "nowhere"), [("engine 1", str(other / "llama-server"))])
    assert [d[0] for d in dirs] == ["settings", "engine 1"]
    found = tools.find(dirs, use_path=False)
    assert found["llama-quantize"].source == "engine 1" and "llama-imatrix" not in found
    first = tools.find([("settings", FAKES), *dirs], use_path=False)
    assert first["llama-quantize"].source == "settings"


def test_a_tool_that_does_not_answer_is_reported_not_hidden(tmp_path: Path):
    broken = tmp_path / "llama-imatrix.py"
    broken.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    found = tools.find([("settings", tmp_path)], use_path=False)
    assert not found["llama-imatrix"].ok and "does not answer" in found["llama-imatrix"].problem
    assert tools.missing_for(found, ("llama-imatrix", "llama-quantize")) == ["llama-imatrix", "llama-quantize"]


def test_nothing_found_is_an_empty_result(tmp_path: Path):
    assert tools.find([("settings", tmp_path)], use_path=False) == {}
    assert tools.find([], use_path=False) == {}
