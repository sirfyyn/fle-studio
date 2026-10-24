from __future__ import annotations

import http.client
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from chat_support import Stack, make_stack, mcp_config
from test_oauth import Fake, start, stop

from fle_studio import mcp_api, tools


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    yield from make_stack(tmp_path, with_engine=False)


def wait_tools(stack: Stack, name: str, timeout: float = 20) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    data: dict[str, Any] = {}
    while time.monotonic() < deadline:
        data = stack.json("GET", "/api/tools")[1]
        if not data["connecting"] and any(t["name"] == name for t in data["tools"]):
            return data
        time.sleep(0.1)
    pytest.fail(f"{name} never appeared: {data.get('servers')}")


def with_fake_app(stack: Stack) -> dict[str, Any]:
    stack.settings({"tools": {"enabled": True, "mcp_enabled": True, "mcp_json": mcp_config()}})
    stack.json("POST", "/api/tools/reload")
    return wait_tools(stack, "fake__show_chart")


# ------------------------------------------------------------------------------------------------ small pieces


def test_tool_meta_is_read_in_all_three_spellings() -> None:
    assert tools.tool_ui_uri({"_meta": {"ui": {"resourceUri": "ui://a/b"}}}) == "ui://a/b"
    assert tools.tool_ui_uri({"_meta": {"ui/resourceUri": "ui://old/one"}}) == "ui://old/one"
    assert tools.tool_ui_uri({"_meta": {"openai/outputTemplate": "ui://widget/x.html"}}) == "ui://widget/x.html"
    assert tools.tool_ui_uri({"_meta": {"ui": {"resourceUri": "https://evil.example/x"}}}) == ""
    assert tools.tool_ui_uri({}) == "" and tools.tool_ui_uri({"_meta": "no"}) == ""


def test_visibility_defaults_to_model_and_app() -> None:
    assert tools.tool_visibility({}) == ["model", "app"]
    assert tools.tool_visibility({"_meta": {"ui": {"visibility": ["app"]}}}) == ["app"]
    assert tools.tool_visibility({"_meta": {"ui": {"visibility": ["model"]}}}) == ["model"]
    assert tools.tool_visibility({"_meta": {"ui": {"visibility": []}}}) == ["model", "app"]


def test_frame_csp_defaults_are_tight() -> None:
    csp = mcp_api.frame_csp({})
    assert "default-src 'none'" in csp and "connect-src 'none'" in csp and "frame-src 'none'" in csp
    assert "'self'" not in csp.replace("frame-ancestors 'self'", "")  # Studio's own origin is never allowed
    assert "form-action 'none'" in csp and "object-src 'none'" in csp
    assert "script-src 'unsafe-inline';" in csp and "unsafe-eval" not in csp


def test_frame_csp_takes_domains_from_the_metadata_and_drops_tricks() -> None:
    meta = {
        "ui": {
            "csp": {
                "connectDomains": ["https://api.example.com", "wss://live.example.com:8443", "http://127.0.0.1:9000"],
                "resourceDomains": [
                    "https://*.cdn.example.com",
                    "https://ok.example; script-src *",
                    "*",
                    "'unsafe-eval'",
                ],
                "frameDomains": ["http://remote.example"],
                "baseUriDomains": "https://x.example",
            }
        }
    }
    csp = mcp_api.frame_csp(meta)
    assert "connect-src https://api.example.com wss://live.example.com:8443 http://127.0.0.1:9000;" in csp
    assert "script-src 'unsafe-inline' https://*.cdn.example.com;" in csp
    assert "frame-src 'none'" in csp and "base-uri 'none'" in csp  # a plain-http remote and a non-list are dropped
    assert "script-src *" not in csp and "unsafe-eval" not in csp


# ------------------------------------------------------------------------------------------------ apps over the API


def test_the_tool_list_marks_app_tools_and_hides_app_only_tools(stack: Stack) -> None:
    data = with_fake_app(stack)
    by_name = {t["name"]: t for t in data["tools"]}
    assert by_name["fake__show_chart"]["ui"] == "ui://fake/chart"
    assert by_name["fake__echo"]["ui"] == ""
    assert "fake__app_bump" not in by_name  # visibility ["app"]: the model never sees it


def test_a_tool_with_an_app_hands_the_whole_result_to_the_page(stack: Stack) -> None:
    with_fake_app(stack)
    call = {"name": "fake__show_chart", "arguments": {"title": "Sales"}, "approved": True}
    status, result = stack.json("POST", "/api/tools/call", call)
    assert status == 200 and result["content"] == "chart of 3 values"
    assert result["app"]["uri"] == "ui://fake/chart" and result["app"]["server"] == "fake"
    assert result["app"]["result"]["structuredContent"] == {"title": "Sales", "values": [3, 5, 8]}
    # a tool without an app carries no extra field
    plain = stack.json("POST", "/api/tools/call", {"name": "fake__echo", "arguments": {"text": "x"}, "approved": True})[
        1
    ]
    assert "app" not in plain


def test_prepare_and_frame_serve_the_app_with_its_own_csp(stack: Stack) -> None:
    with_fake_app(stack)
    status, prep = stack.json("POST", "/api/apps/prepare", {"name": "fake__show_chart"})
    assert status == 200 and prep["prefers_border"] is True and prep["uri"] == "ui://fake/chart"
    status, headers, body = stack.call("GET", prep["url"])
    assert status == 200 and headers["content-type"].startswith("text/html")
    assert b"ui/initialize" in body
    csp = headers["content-security-policy"]
    assert csp.startswith("default-src 'none'") and "connect-src 'none'" in csp and "frame-ancestors 'self'" in csp
    assert "x-frame-options" not in headers  # Studio's pages are DENY; the frame must be embeddable by the page
    assert headers["referrer-policy"] == "no-referrer" and headers["x-content-type-options"] == "nosniff"
    # the page's own CSP still forbids everything but its own files
    index = stack.call("GET", "/")[1]
    assert "frame-ancestors 'none'" in index["content-security-policy"] and index["x-frame-options"] == "DENY"


def test_prepare_refuses_tools_without_an_app_and_unknown_frames(stack: Stack) -> None:
    with_fake_app(stack)
    assert stack.json("POST", "/api/apps/prepare", {"name": "fake__echo"})[0] == 404
    assert stack.json("POST", "/api/apps/prepare", {"name": "calculator"})[0] == 404
    assert stack.call("GET", "/app-frame/" + "a" * 32)[0] == 404
    assert stack.json("POST", "/api/apps/call", {"frame": "nope", "name": "echo"})[0] == 404


def test_an_app_calls_its_own_server_and_writing_calls_need_approval(stack: Stack) -> None:
    with_fake_app(stack)
    frame = stack.json("POST", "/api/apps/prepare", {"name": "fake__show_chart"})[1]["frame"]
    call = {"frame": frame, "name": "app_bump", "arguments": {}}
    status, res = stack.json("POST", "/api/apps/call", call)
    assert status == 409 and "approval" in res["error"]["message"]  # same rule as for the model: untrusted -> ask
    assert stack.json("POST", "/api/apps/call", {**call, "approved": "yes"})[0] == 409
    status, res = stack.json("POST", "/api/apps/call", {**call, "approved": True})
    assert status == 200 and res["result"]["structuredContent"] == {"count": 1}
    stack.settings({"tools": {"mcp_trusted": ["fake"]}})
    status, res = stack.json("POST", "/api/apps/call", call)
    assert status == 200 and res["result"]["structuredContent"] == {"count": 2}  # trusted: no question
    # a tool the server does not have, and a tool of another server, are not reachable
    assert stack.json("POST", "/api/apps/call", {**call, "name": "no_such_tool"})[0] == 404
    assert stack.json("POST", "/api/apps/call", {**call, "name": "calculator"})[0] == 404
    assert (
        stack.json("POST", "/api/apps/call", {**call, "name": "fake__echo"})[0] == 404
    )  # the wire name is not the tool's


def test_app_tools_obey_their_visibility() -> None:
    box = tools.ToolBox.__new__(tools.ToolBox)

    class Srv:
        status = "ready"
        tools = [
            {"name": "only_model", "_meta": {"ui": {"visibility": ["model"]}}},
            {"name": "only_app", "_meta": {"ui": {"visibility": ["app"]}}},
            {"name": "both"},
        ]

    class Manager:
        def get(self, name: str) -> Any:
            return Srv() if name == "s" else None

    box.mcp = Manager()  # type: ignore[assignment]
    assert box.app_tool("s", "only_app")["name"] == "only_app" and box.app_tool("s", "both")["name"] == "both"
    with pytest.raises(KeyError):
        box.app_tool("s", "only_model")  # an app may not call a tool meant for the model alone
    with pytest.raises(KeyError):
        box.app_tool("other", "both")


def test_a_resource_that_is_not_an_app_is_refused(stack: Stack) -> None:
    with_fake_app(stack)
    server = stack.studio.__dict__["toolbox"].mcp.get("fake")
    from fle_studio import mcp  # noqa: PLC0415

    with pytest.raises(mcp.McpError, match="ui://"):
        server.read_resource("file:///etc/passwd")
    with pytest.raises(mcp.McpError):
        server.read_resource("ui://fake/missing")


# ------------------------------------------------------------------------------------------------ sign-in over the API


@pytest.fixture
def remote(tmp_path: Path) -> Iterator[Fake]:
    fake = start("--oauth")
    try:
        yield fake
    finally:
        stop(fake)


def browser_get(url: str) -> tuple[int, str]:
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname or "", u.port, timeout=10)
    conn.request("GET", u.path + ("?" + u.query if u.query else ""))
    resp = conn.getresponse()
    resp.read()
    where = resp.getheader("Location") or ""
    conn.close()
    return resp.status, where


def server_row(stack: Stack, name: str = "web") -> dict[str, Any]:
    return next(s for s in stack.json("GET", "/api/tools")[1]["servers"] if s["name"] == name)


def wait_status(stack: Stack, status: str, timeout: float = 20) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    row: dict[str, Any] = {}
    while time.monotonic() < deadline:
        row = server_row(stack)
        if row["status"] == status:
            return row
        time.sleep(0.1)
    pytest.fail(f"never became {status}: {row}")


def test_sign_in_and_out_through_the_api(stack: Stack, remote: Fake, tmp_path: Path) -> None:
    stack.settings(
        {
            "tools": {
                "enabled": True,
                "mcp_enabled": True,
                "mcp_json": json.dumps({"mcpServers": {"web": {"url": remote.url}}}),
            }
        }
    )
    row = wait_status(stack, "auth_required")
    assert row["auth"]["remote"] and row["auth"]["required"] and row["auth"]["signed_in"] is False

    status, started = stack.json("POST", "/api/mcp/signin", {"server": "web"})
    assert status == 200 and started["auth_url"].startswith(f"http://127.0.0.1:{remote.port}/authorize?")
    status, back = browser_get(started["auth_url"])
    assert status == 302 and back.startswith(f"http://127.0.0.1:{stack.port}/oauth/callback?")
    status, _, page = stack.call("GET", urlparse(back).path + "?" + urlparse(back).query)  # the browser lands on Studio
    assert status == 200 and b"Signed in" in page
    row = wait_status(stack, "ready")
    assert row["auth"]["signed_in"] is True and row["tools"] >= 5
    assert "access_token" not in json.dumps(stack.json("GET", "/api/tools")[1])
    assert "access_token" not in json.dumps(stack.json("GET", "/api/settings")[1])
    assert remote.events("register")[0]["redirects"] == [f"http://127.0.0.1:{stack.port}/oauth/callback"]

    status, res = stack.json("POST", "/api/mcp/signout", {"server": "web"})
    assert status == 200 and res["had_token"] is True
    row = wait_status(stack, "auth_required")
    assert row["auth"]["signed_in"] is False
    assert not list((stack.studio.data_dir / "oauth").glob("*.json"))


def test_a_callback_nobody_asked_for_is_refused_and_escaped(stack: Stack) -> None:
    status, _, page = stack.call("GET", "/oauth/callback?state=nope&code=x")
    assert status == 400 and b"not known" in page
    status, _, page = stack.call("GET", "/oauth/callback?state=nope&error=%3Cscript%3Ealert(1)%3C/script%3E")
    assert status == 400 and b"<script>" not in page


def test_sign_in_for_a_stdio_server_or_an_unknown_one_fails(stack: Stack) -> None:
    with_fake_app(stack)
    assert stack.json("POST", "/api/mcp/signin", {"server": "fake"})[0] == 404
    assert stack.json("POST", "/api/mcp/signin", {"server": "ghost"})[0] == 404
