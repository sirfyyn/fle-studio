from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import pytest
from chat_support import FAKE_MCP, clean_env, free_port

from fle_studio import mcp


def stdio_spec(*args: str) -> dict[str, Any]:
    return {"transport": "stdio", "command": sys.executable, "args": [str(FAKE_MCP), *args], "env": {}, "cwd": ""}


@pytest.fixture
def server() -> Iterator[mcp.McpServer]:
    srv = mcp.McpServer("fake", stdio_spec(), 20)
    srv.connect()
    try:
        yield srv
    finally:
        srv.close()


def test_parse_servers_reads_both_kinds() -> None:
    text = json.dumps(
        {
            "mcpServers": {
                "local": {"command": "npx", "args": ["-y", "pkg"], "env": {"A": 1}},
                "remote": {"url": "https://example.org/mcp", "headers": {"Authorization": "Bearer x"}},
                "off": {"command": "x", "disabled": True},
            }
        }
    )
    servers, problems = mcp.parse_servers(text)
    assert problems == {}
    assert set(servers) == {"local", "remote"}
    assert servers["local"]["env"] == {"A": "1"} and servers["remote"]["transport"] == "http"


def test_parse_servers_names_each_problem() -> None:
    assert mcp.parse_servers("") == ({}, {})
    assert "not valid JSON" in mcp.parse_servers("{oops")[1][""]
    servers, problems = mcp.parse_servers(
        json.dumps(
            {
                "a": {"type": "sse", "url": "http://x/sse"},
                "b": {"url": "ftp://x"},
                "c": {},
                "d": {"command": "x", "args": "no"},
                "e": 5,
                "ok": {"command": "x"},
            }
        )
    )
    assert set(servers) == {"ok"} and set(problems) == {"a", "b", "c", "d", "e"}


def test_stdio_connect_lists_tools(server: mcp.McpServer) -> None:
    assert server.status == "ready", server.error
    assert {t["name"] for t in server.tools} >= {"echo", "add", "write_note"}
    assert server.info["name"] == "fake-mcp"


def test_stdio_call_and_tool_errors(server: mcp.McpServer) -> None:
    assert server.call("echo", {"text": "hi"}) == ("echo: hi", False)
    assert server.call("add", {"a": 2, "b": 3}) == ("5", False)
    assert server.call("fail", {}) == ("boom", True)
    with pytest.raises(mcp.McpError):
        server.call("no_such_tool", {})


def test_paging_collects_every_tool() -> None:
    srv = mcp.McpServer("paged", stdio_spec(), 20)
    os.environ["FAKE_MCP_PAGE"] = "2"
    try:
        srv.connect()
    finally:
        del os.environ["FAKE_MCP_PAGE"]
    try:
        assert srv.status == "ready" and len(srv.tools) >= 5
    finally:
        srv.close()


def test_a_crash_is_an_error_and_the_server_starts_again(server: mcp.McpServer) -> None:
    with pytest.raises(mcp.McpError):
        server.call("crash", {})
    # the next call finds the process gone and starts it once more
    assert server.call("echo", {"text": "again"}) == ("echo: again", False)


def test_a_missing_program_is_an_error_not_an_exception() -> None:
    srv = mcp.McpServer(
        "bad", {"transport": "stdio", "command": "definitely-not-a-program-xyz", "args": [], "env": {}, "cwd": ""}
    )
    srv.connect()
    assert srv.status == "error" and srv.error
    with pytest.raises(mcp.McpError):
        srv.call("echo", {})


def test_close_ends_the_process(server: mcp.McpServer) -> None:
    transport = server.transport
    assert isinstance(transport, mcp.StdioTransport)
    proc = transport.proc  # type: ignore[attr-defined]
    server.close()
    deadline = time.monotonic() + 10
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None


def test_result_text_names_media_and_cuts_long_text() -> None:
    text, err = mcp.result_text(
        {"content": [{"type": "text", "text": "a"}, {"type": "image", "mimeType": "image/png"}], "isError": True}
    )
    assert err is True and text.startswith("a\n[image image/png")
    assert mcp.result_text({"structuredContent": {"x": 1}})[0] == '{"x": 1}'
    long = mcp.result_text({"content": [{"type": "text", "text": "z" * (mcp.MAX_RESULT_CHARS + 10)}]})[0]
    assert "cut off" in long


def test_manager_syncs_and_closes() -> None:
    manager = mcp.McpManager()
    try:
        manager.sync({"one": stdio_spec()}, 20)
        assert manager.wait_idle(20)
        assert manager.get("one").status == "ready"  # type: ignore[union-attr]
        manager.sync({}, 20)
        assert manager.servers() == []
    finally:
        manager.close_all()


# ------------------------------------------------------------------------------------------------ Streamable HTTP


def start_http(*extra: str, token: str | None = None) -> tuple[subprocess.Popen[bytes], str]:
    port = free_port()
    env = clean_env({"FAKE_MCP_TOKEN": token} if token else {})
    proc = subprocess.Popen(
        [sys.executable, str(FAKE_MCP), "--http", str(port), *extra],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stderr is not None
    proc.stderr.readline()  # "fake mcp listening on …"
    return proc, f"http://127.0.0.1:{port}/mcp"


@pytest.mark.parametrize("extra", [(), ("--sse",)], ids=["json", "sse"])
def test_http_transport(extra: tuple[str, ...]) -> None:
    proc, url = start_http(*extra)
    srv = mcp.McpServer("web", {"transport": "http", "url": url, "headers": {}}, 20)
    try:
        srv.connect()
        assert srv.status == "ready", srv.error
        assert srv.call("echo", {"text": "over http"}) == ("echo: over http", False)
        assert srv.call("fail", {}) == ("boom", True)
    finally:
        srv.close()
        proc.kill()
        proc.wait(timeout=10)
        if proc.stderr:
            proc.stderr.close()


def test_http_sends_the_configured_headers() -> None:
    proc, url = start_http(token="s3cret")
    try:
        bad = mcp.McpServer("web", {"transport": "http", "url": url, "headers": {}}, 20)
        bad.connect()
        assert bad.status == "error" and "401" in bad.error
        good = mcp.McpServer(
            "web", {"transport": "http", "url": url, "headers": {"Authorization": "Bearer s3cret"}}, 20
        )
        good.connect()
        try:
            assert good.status == "ready", good.error
        finally:
            good.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)
        if proc.stderr:
            proc.stderr.close()
