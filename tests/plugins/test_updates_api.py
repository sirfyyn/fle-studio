"""The /api/update endpoints on a real server, with a fake GitHub on a loopback port."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fake_github import FakeGitHub, release

from fle_studio.server import Studio, make_server
from fle_studio_updater import updates


class Page:
    def __init__(self, studio: Studio, port: int) -> None:
        self.studio, self.port = studio, port

    def call(self, method: str, path: str, body: dict[str, Any] | None = None, token: bool = True) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-FLE-Token"] = self.studio.token
        conn.request(
            method,
            path,
            body=json.dumps(body if body is not None else {}) if method == "POST" else None,
            headers=headers,
        )
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"null")
        conn.close()
        return resp.status, data


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Page, FakeGitHub]]:
    with FakeGitHub([release("v9.0.0", "# New\n\n- one\n\n<script>x</script>")]) as gh:
        monkeypatch.setenv("FLE_STUDIO_GITHUB_API", gh.url)
        monkeypatch.setattr(updates, "find_uv", lambda: None)
        st = Studio(tmp_path / "home")
        server = make_server(st, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
        yield Page(st, int(server.server_address[1])), gh
        server.shutdown()
        server.server_close()


def test_starting_the_studio_asks_nobody(env: tuple[Page, FakeGitHub]) -> None:
    page, gh = env
    status, view = page.call("GET", "/api/update")
    assert status == 200 and gh.requests == []
    assert view["enabled"] is True and view["app"] is None and view["current"]
    assert set(view["install"]) >= {"kind", "source", "service"}


def test_check_dismiss_and_the_shape_of_the_answer(env: tuple[Page, FakeGitHub]) -> None:
    page, gh = env
    status, view = page.call("POST", "/api/update/check")
    assert status == 200 and view["app"]["version"] == "9.0.0" and view["app"]["dismissed"] is False
    assert "<script>" in view["app"]["notes"]  # kept as text; the page shows it as text
    assert view["app"]["url"] == "https://github.com/sirfyyn/fle-studio/releases/tag/v9.0.0"
    assert {"can_auto", "reason", "command"} <= set(view["app"])
    assert len(gh.requests) == 1
    status, view = page.call("POST", "/api/update/dismiss", {"version": "9.0.0"})
    assert status == 200 and view["app"]["dismissed"] is True
    assert page.call("GET", "/api/update")[1]["app"]["dismissed"] is True


def test_switching_off_stops_the_asking(env: tuple[Page, FakeGitHub]) -> None:
    page, gh = env
    status, _ = page.call("POST", "/api/settings", {"updates": {"check": False}})
    assert status == 200
    status, body = page.call("POST", "/api/update/check")
    assert status == 409 and "off" in json.dumps(body).lower()
    assert gh.requests == []
    assert page.call("GET", "/api/update")[1]["enabled"] is False


def test_apply_is_refused_where_the_install_cannot_update_itself(env: tuple[Page, FakeGitHub]) -> None:
    page, _ = env
    page.call("POST", "/api/update/check")
    status, body = page.call("POST", "/api/update/apply", {"tag": "v9.0.0"})
    assert status == 409 and body.get("error")
    assert page.call("GET", "/api/update")[1]["run"]["state"] == "idle"


def test_changes_need_the_page_token(env: tuple[Page, FakeGitHub]) -> None:
    page, gh = env
    for path in ("check", "dismiss", "apply", "ack"):
        assert page.call("POST", f"/api/update/{path}", token=False)[0] == 403
    assert gh.requests == []
