"""One browser connection carries many requests (HTTP/1.1 keep-alive). The page sends `{}` with every POST, also to
handlers that read no body; whatever a handler leaves must not become the start of the next request."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from fle_studio.server import Studio, make_server


@pytest.fixture
def studio(tmp_path: Path) -> Iterator[tuple[Studio, int]]:
    st = Studio(tmp_path / "home")
    server = make_server(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield st, int(server.server_address[1])
    server.shutdown()
    server.server_close()


def test_an_unread_body_does_not_spill_into_the_next_request(studio: tuple[Studio, int]) -> None:
    st, port = studio
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"X-FLE-Token": st.token, "Content-Type": "application/json"}
    for _ in range(2):
        conn.request("POST", "/api/engine/stop", body="{}", headers=headers)  # the handler reads no body
        resp = conn.getresponse()
        assert resp.status == 200, resp.read()
        assert json.loads(resp.read())["state"] == "stopped"
        conn.request("GET", "/api/ping")
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read())["app"] == "fle-studio"
    conn.close()


def test_a_refused_post_leaves_the_connection_clean(studio: tuple[Studio, int]) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", studio[1], timeout=10)
    conn.request("POST", "/api/engine/stop", body='{"x": 1}', headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 403  # no page token
    resp.read()
    conn.request("GET", "/api/ping")
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()
    conn.close()
