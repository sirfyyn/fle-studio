"""The core runs without the plugins that ship with FLE Studio (FLE_STUDIO_NO_BUNDLED=1).

The CI runs the whole core suite like that too (see .github/workflows/ci.yml): everything in tests/ except
tests/plugins/ has to pass with no bundled plugin at all.
"""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fle_studio import plugins
from fle_studio.server import Studio, make_server


@pytest.fixture
def core(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Studio, int]]:
    monkeypatch.setenv(plugins.NO_BUNDLED, "1")
    studio = Studio(tmp_path / "home")
    server = make_server(studio, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield studio, int(server.server_address[1])
    server.shutdown()
    server.server_close()
    studio.plugins.stop()


def call(studio: Studio, port: int, method: str, path: str, body: Any = None) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        method,
        path,
        body=json.dumps(body or {}) if method == "POST" else None,
        headers={"Content-Type": "application/json", "X-FLE-Token": studio.token},
    )
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        return resp.status, json.loads(raw or b"null")
    except ValueError:
        return resp.status, raw


def test_no_bundled_plugin_is_found_or_loaded(core: tuple[Studio, int]) -> None:
    studio, port = core
    status, view = call(studio, port, "GET", "/api/plugins")
    assert status == 200 and view["plugins"] == [] and view["error"] == ""
    assert studio.plugins.loaded == {}


def test_the_proxy_streams_and_shows_the_speed_without_a_recorder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chat_support import make_stack

    monkeypatch.setenv(plugins.NO_BUNDLED, "1")
    body = {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16}
    for stack in make_stack(tmp_path):
        assert stack.studio.plugins.hooks("proxy_record") == []
        stack.studio.live.touch()  # a page is open
        status, _, raw = stack.call("POST", "/v1/chat/completions", {**body, "stream": True})
        assert status == 200 and b"data:" in raw and b"[DONE]" in raw
        status, live = stack.json("GET", "/api/live")
        assert status == 200 and live["last_response"] and live["last_response"]["predicted_n"], "speed from the tail"
        status, answer = stack.chat(body)  # not streamed
        assert status == 200 and answer["choices"]


def test_the_core_pages_and_calls_answer_without_plugins(core: tuple[Studio, int]) -> None:
    studio, port = core
    for path in ("/api/state", "/api/engine", "/api/live", "/api/settings", "/api/chats", "/api/tools", "/api/models"):
        status, _ = call(studio, port, "GET", path)
        assert status == 200, path
    status, body = call(studio, port, "GET", "/")
    assert status == 200 and b"fle-token" in body
    # what a plugin would have added is simply not there
    for gone in (
        "/api/engine/switch",
        "/api/forge",
        "/api/update",
        "/api/livechat",
        "/api/capture",
        "/api/remote",
        "/api/hosts",
        "/h/abc/api/state",
    ):
        assert call(studio, port, "GET", gone)[0] == 404, gone


def test_a_server_beyond_this_machine_speaks_plain_http_without_the_remote_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(plugins.NO_BUNDLED, "1")
    studio = Studio(tmp_path / "home")
    server = make_server(studio, "127.0.0.1", 0, tls=True)  # TLS asked for, nobody to provide it
    try:
        assert studio.listen["tls"] is False and studio.tls_fingerprint == ""
    finally:
        server.server_close()
        studio.plugins.stop()
