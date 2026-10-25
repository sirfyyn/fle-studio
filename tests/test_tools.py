from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from chat_support import mcp_config

from fle_studio import tools
from fle_studio.config import Config


def make_box(tmp_path: Path, **settings: Any) -> tools.ToolBox:
    config = Config(tmp_path / "settings.json")
    if settings:
        config.update({"tools": settings})
    return tools.ToolBox(config)


@pytest.fixture
def box(tmp_path: Path) -> Iterator[tools.ToolBox]:
    b = make_box(tmp_path)
    try:
        yield b
    finally:
        b.close()


# ------------------------------------------------------------------------------------------------ calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", "7"),
        ("(12.5 + 7) * 3 / 4", "14.625"),
        ("2^10", "1024"),
        ("7 // 2 + 7 % 4", "6"),
        ("sqrt(16) + abs(-3)", "7"),
        ("3 × 4 − 2", "10"),
        ("pi", "3.14159265359"),
        ("-(2 ** 3)", "-8"),
    ],
)
def test_calculator(expression: str, expected: str) -> None:
    assert tools.calculator({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "().__class__",
        "[1,2][0]",
        "x + 1",
        "1 / 0",
        "9 ** 9 ** 9",
        "10 ** 100000",
        "factorial(5000)",
        "1 +",
        "sqrt(-1)",
        "-" * 60 + "1",
        "1" * 600,
        "lambda: 1",
        "1 if 1 else 2",
        "'a' * 5",
    ],
)
def test_calculator_refuses(expression: str) -> None:
    with pytest.raises(tools.ToolError):
        tools.calculator({"expression": expression})


def test_clock() -> None:
    assert "UTC" in tools.clock({"timezone": "UTC"})
    assert tools.clock({})
    with pytest.raises(tools.ToolError):
        tools.clock({"timezone": "Not/AZone"})


# ------------------------------------------------------------------------------------------------ files


@pytest.fixture
def shared(tmp_path: Path) -> Path:
    root = tmp_path / "shared"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"hello\nworld\n")
    (root / "sub" / "b.txt").write_text("deep", encoding="utf-8")
    (root / "bin.dat").write_bytes(b"\x00\x01\x02")
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    return root


def test_files_list_and_read(shared: Path) -> None:
    listing = tools.list_files(str(shared), {})
    assert "sub/" in listing and "a.txt" in listing
    assert tools.read_file(str(shared), {"path": "a.txt"}) == "hello\nworld\n"
    assert tools.read_file(str(shared), {"path": "sub/b.txt"}) == "deep"
    assert tools.list_files(str(shared), {"path": "sub"}).startswith("sub:")


def test_files_read_in_pieces(shared: Path) -> None:
    (shared / "long.txt").write_text("x" * 50, encoding="utf-8")
    first = tools.read_file(str(shared), {"path": "long.txt", "max_chars": 10})
    assert first.startswith("x" * 10) and "offset=10" in first
    assert tools.read_file(str(shared), {"path": "long.txt", "offset": 40, "max_chars": 100}) == "x" * 10


@pytest.mark.parametrize("path", ["../secret.txt", "sub/../../secret.txt", "..", "a\x00.txt"])
def test_files_cannot_leave_the_folder(shared: Path, path: str) -> None:
    with pytest.raises(tools.ToolError):
        tools.read_file(str(shared), {"path": path})


def test_files_absolute_path_outside_is_refused(shared: Path, tmp_path: Path) -> None:
    with pytest.raises(tools.ToolError):
        tools.read_file(str(shared), {"path": str(tmp_path / "secret.txt")})
    assert tools.read_file(str(shared), {"path": str(shared / "a.txt")}).startswith("hello")


def test_files_symlink_out_is_refused(shared: Path, tmp_path: Path) -> None:
    link = shared / "link.txt"
    try:
        os.symlink(tmp_path / "secret.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available here")
    with pytest.raises(tools.ToolError):
        tools.read_file(str(shared), {"path": "link.txt"})


def test_files_problems_are_messages(shared: Path) -> None:
    with pytest.raises(tools.ToolError, match="binary"):
        tools.read_file(str(shared), {"path": "bin.dat"})
    with pytest.raises(tools.ToolError, match="no such file"):
        tools.read_file(str(shared), {"path": "missing.txt"})
    with pytest.raises(tools.ToolError, match="No folder"):
        tools.read_file("", {"path": "a.txt"})
    with pytest.raises(tools.ToolError):
        tools.read_file(str(shared), {"path": "a.txt", "offset": "x"})


# ------------------------------------------------------------------------------------------------ web search


DUCK_HTML = """
<div class="result">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fone&rut=x">First  <b>hit</b></a>
<a class="result__snippet" href="x">About the   first thing</a></div>
<div class="result"><a class="result__a" href="https://example.org/two">Second</a></div>
"""


def test_duckduckgo_parser() -> None:
    rows = tools.parse_duckduckgo(DUCK_HTML)
    assert rows[0] == {"title": "First hit", "url": "https://example.org/one", "snippet": "About the first thing"}
    assert rows[1]["url"] == "https://example.org/two" and rows[1]["snippet"] == ""


def test_searxng_parser() -> None:
    rows = tools.parse_searxng(
        json.dumps({"results": [{"title": "T", "url": "http://x", "content": "C"}, {"title": "no url"}]})
    )
    assert rows == [{"title": "T", "url": "http://x", "snippet": "C"}]
    with pytest.raises(tools.ToolError):
        tools.parse_searxng("<html>")


@pytest.fixture
def searx() -> Iterator[str]:
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            seen.append(self.path)
            body = json.dumps(
                {"results": [{"title": f"R{i}", "url": f"http://x/{i}", "content": "c"} for i in range(8)]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/search"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_web_search_against_a_local_searxng(searx: str) -> None:
    out = tools.web_search(searx, {"query": "fish & chips", "max_results": 3})
    assert out.count("http://x/") == 3 and "R0" in out
    with pytest.raises(tools.ToolError):
        tools.web_search(searx, {"query": ""})
    with pytest.raises(tools.ToolError, match="http"):
        tools.web_search("file:///etc/passwd", {"query": "x"})


# ------------------------------------------------------------------------------------------------ the set


def names(box: tools.ToolBox) -> set[str]:
    return {t.name for t in box.tools()}


def test_defaults_are_careful(box: tools.ToolBox) -> None:
    assert names(box) == {"calculator", "get_current_time"}  # no files without a folder, no web, no MCP


def test_each_tool_can_be_turned_off(tmp_path: Path) -> None:
    box = make_box(tmp_path, calculator=False)
    assert names(box) == {"get_current_time"}
    assert make_box(tmp_path, enabled=False).tools() == []


def test_file_tools_need_a_real_folder(tmp_path: Path, shared: Path) -> None:
    assert "read_file" not in names(make_box(tmp_path, files=True, files_dir=""))
    assert "read_file" not in names(make_box(tmp_path, files=True, files_dir=str(tmp_path / "nope")))
    assert {"read_file", "list_files"} <= names(make_box(tmp_path, files=True, files_dir=str(shared)))


def test_web_search_asks_before_each_search(tmp_path: Path) -> None:
    box = make_box(tmp_path, web_search=True)
    tool = next(t for t in box.tools() if t.name == "web_search")
    assert tool.confirm is True
    with pytest.raises(tools.NeedsApproval):
        box.call("web_search", {"query": "x"})
    box = make_box(tmp_path, web_search=True, web_search_confirm=False)
    assert next(t for t in box.tools() if t.name == "web_search").confirm is False


def test_call_runs_a_tool_and_reports_errors(box: tools.ToolBox) -> None:
    ok = box.call("calculator", {"expression": "6*7"})
    assert ok["content"] == "42" and ok["is_error"] is False and ok["ms"] >= 0
    assert box.call("calculator", '{"expression": "1+1"}')["content"] == "2"  # arguments as JSON text
    bad = box.call("calculator", {"expression": "1/0"})
    assert bad["is_error"] is True and "zero" in bad["content"]
    assert box.call("calculator", "{broken")["is_error"] is True
    assert box.call("calculator", "[1]")["is_error"] is True
    with pytest.raises(KeyError):
        box.call("rm_rf", {})


def test_openai_shape(box: tools.ToolBox) -> None:
    spec = box.tools()[0].openai()
    assert spec["type"] == "function" and spec["function"]["parameters"]["type"] == "object"


# ------------------------------------------------------------------------------------------------ MCP tools


def test_mcp_tools_ask_unless_the_server_is_trusted(tmp_path: Path) -> None:
    box = make_box(tmp_path, mcp_enabled=True, mcp_json=mcp_config())
    try:
        box.sync_mcp()
        assert box.mcp.wait_idle(20)
        listed = {t.name: t for t in box.tools()}
        assert "fake__echo" in listed and "fake__write_note" in listed
        assert listed["fake__echo"].confirm is True and listed["fake__echo"].server == "fake"
        with pytest.raises(tools.NeedsApproval):
            box.call("fake__write_note", {"text": "x"})
        done = box.call("fake__write_note", {"text": "x"}, approved=True)
        assert done["content"] == "saved note 1" and done["is_error"] is False
        failed = box.call("fake__fail", {}, approved=True)
        assert failed == {**failed, "content": "boom", "is_error": True}
        box.config.update({"tools": {"mcp_trusted": ["fake"]}})
        assert box.call("fake__echo", {"text": "t"})["content"] == "echo: t"  # trusted: no question
    finally:
        box.close()


def test_mcp_off_means_no_process(tmp_path: Path) -> None:
    box = make_box(tmp_path, mcp_enabled=False, mcp_json=mcp_config())
    try:
        assert not any(t.source == "mcp" for t in box.tools())
        assert box.mcp.servers() == []
    finally:
        box.close()


def test_status_reports_servers_and_problems(tmp_path: Path) -> None:
    box = make_box(
        tmp_path,
        mcp_enabled=True,
        mcp_json=json.dumps({"mcpServers": {"bad": {}, "fake": json.loads(mcp_config())["mcpServers"]["fake"]}}),
    )
    try:
        box.sync_mcp()
        assert box.mcp.wait_idle(20)
        status = box.status()
        by_name = {s["name"]: s for s in status["servers"]}
        assert by_name["fake"]["status"] == "ready" and by_name["bad"]["status"] == "error"
        assert status["connecting"] is False
    finally:
        box.close()


def test_tools_come_sorted_by_name(tmp_path: Path) -> None:
    box = make_box(tmp_path, calculator=True, datetime=True, mcp_enabled=True, mcp_json=mcp_config())
    try:
        box.sync_mcp()
        assert box.mcp.wait_idle(20)
        names = [t.name for t in box.tools()]
        assert names == sorted(names) and "calculator" in names and "fake__echo" in names
        assert [t["name"] for t in box.status()["tools"]] == names
    finally:
        box.close()


def test_a_late_server_keeps_its_last_known_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fle_studio import mcp as mcp_module

    data = tmp_path / "data"
    config = Config(tmp_path / "settings.json")
    config.update({"tools": {"mcp_enabled": True, "mcp_json": mcp_config()}})
    first = tools.ToolBox(config, data)
    try:
        first.sync_mcp()
        assert first.mcp.wait_idle(20)
        before = [(t.name, t.parameters) for t in first.tools()]
        assert any(n == "fake__echo" for n, _ in before) and not any(t.stale for t in first.tools())
    finally:
        first.close()

    # A new start where the server is slow to become ready: it never leaves "idle" here.
    monkeypatch.setattr(mcp_module.McpServer, "connect", lambda self: None)
    second = tools.ToolBox(config, data)
    try:
        second.sync_mcp()
        late = second.tools()
        assert [(t.name, t.parameters) for t in late] == before, "the same tools in the same order"
        assert all(t.stale for t in late if t.source == "mcp")
        # a different configuration is another server: nothing is carried over
        config.update({"tools": {"mcp_json": mcp_config("--other")}})
        assert not any(t.source == "mcp" for t in second.tools())
    finally:
        second.close()


def test_wire_names_are_safe_and_unique() -> None:
    taken: set[str] = set()
    first = tools._wire_name("my server", "do.it!", taken)
    taken.add(first)
    assert first == "my_server__do_it_" and tools._wire_name("my server", "do.it!", taken) == first + "_2"
    assert len(tools._wire_name("s" * 100, "t" * 100, set())) <= 64


# ------------------------------------------------------------------------------------------------ the trap


def test_tools_and_forced_format_never_travel_together() -> None:
    tool = [{"type": "function", "function": {"name": "x", "parameters": {"type": "object"}}}]
    for field in ("response_format", "json_schema", "grammar"):
        value: Any = (
            {"type": "json_object"}
            if field == "response_format"
            else ({"type": "object"} if field == "json_schema" else "root ::= x")
        )
        with pytest.raises(ValueError, match=field):
            tools.check_tools_and_format({"tools": tool, field: value})
    # harmless combinations
    tools.check_tools_and_format({"tools": tool})
    tools.check_tools_and_format({"tools": tool, "response_format": {"type": "text"}})
    tools.check_tools_and_format({"tools": tool, "tool_choice": "none", "response_format": {"type": "json_object"}})
    tools.check_tools_and_format({"response_format": {"type": "json_object"}})
    tools.check_tools_and_format({"tools": [], "grammar": "root ::= x"})
