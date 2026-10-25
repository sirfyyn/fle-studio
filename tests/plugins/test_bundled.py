"""The plugins that ship with FLE Studio are found by the entry points, are on by default and can be turned off."""

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

SHIPPED = {"forge", "switch", "updater"}


@pytest.fixture
def running(tmp_path: Path) -> Iterator[tuple[Studio, int]]:
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


def test_the_package_ships_its_plugins() -> None:
    found = {f.id: f for f in plugins.discover()}
    assert SHIPPED <= set(found)
    assert all(found[i].bundled and found[i].package == "fle-studio" for i in SHIPPED)


def test_they_are_on_by_default_and_each_can_be_turned_off(running: tuple[Studio, int]) -> None:
    studio, port = running
    _, view = call(studio, port, "GET", "/api/plugins")
    by_id = {p["id"]: p for p in view["plugins"]}
    for pid in SHIPPED:
        assert by_id[pid]["bundled"] and by_id[pid]["enabled"] and by_id[pid]["loaded"] and not by_id[pid]["error"]

    status, sw = call(studio, port, "GET", "/api/engine/switch")
    assert status == 200 and sw == {"available": False}  # no switch_url set

    status, view = call(studio, port, "POST", "/api/plugins/switch/disable")
    assert status == 200 and not {p["id"]: p for p in view["plugins"]}["switch"]["loaded"]
    assert call(studio, port, "GET", "/api/engine/switch")[0] == 404
    assert call(studio, port, "GET", "/api/state")[0] == 200, "the rest goes on"

    call(studio, port, "POST", "/api/plugins/switch/enable")
    assert call(studio, port, "GET", "/api/engine/switch")[0] == 200


def test_the_update_plugin_brings_its_routes_and_its_restart_hook(running: tuple[Studio, int]) -> None:
    studio, port = running
    status, data = call(studio, port, "GET", "/api/update")
    assert status == 200 and "install" in data and "run" in data
    assert studio.plugins.loaded["updater"]._boot == "boot.js"
    call(studio, port, "POST", "/api/plugins/updater/disable")
    assert call(studio, port, "GET", "/api/update")[0] == 404
    assert call(studio, port, "GET", "/api/state")[0] == 200


def test_forge_is_a_page_and_routes_that_come_and_go_with_the_plugin(running: tuple[Studio, int]) -> None:
    studio, port = running
    _, view = call(studio, port, "GET", "/api/plugins")
    forge = {p["id"]: p for p in view["plugins"]}["forge"]
    assert forge["page"] == {"title": "Forge", "icon": "forge", "script": "forge.js", "path": "/forge"}
    status, body = call(studio, port, "GET", "/plugins/forge/forge.js")
    assert status == 200 and b"export function mount" in body
    assert call(studio, port, "GET", "/plugins/forge/forge.css")[0] == 200
    status, data = call(studio, port, "GET", "/api/forge")
    assert status == 200 and {t["name"] for t in data["tools"]} >= {"llama-quantize"}
    assert call(studio, port, "GET", "/forge")[0] == 200, "the page path falls back to the app"

    call(studio, port, "POST", "/api/plugins/forge/disable")
    assert call(studio, port, "GET", "/api/forge")[0] == 404
    assert call(studio, port, "GET", "/plugins/forge/forge.js")[0] == 404
    assert call(studio, port, "GET", "/api/models")[0] == 200
