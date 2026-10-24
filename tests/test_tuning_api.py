"""The routes of tuning_api.py through a real FLE Studio server."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fle_studio.server import Studio, make_server


class Client:
    def __init__(self, studio: Studio, port: int) -> None:
        self.studio, self.port = studio, port

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = {"X-FLE-Token": self.studio.token, "Content-Type": "application/json"}
        conn.request(method, path, body=json.dumps(body if body is not None else {}), headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"null")
        conn.close()
        return resp.status, data


@pytest.fixture
def api(tmp_path: Path) -> Iterator[Client]:
    st = Studio(tmp_path / "home")
    server = make_server(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield Client(st, int(server.server_address[1]))
    server.shutdown()
    server.server_close()


def test_flag_table_and_explanations(api: Client) -> None:
    status, table = api.call("GET", "/api/flags")
    assert status == 200 and "-t" in table["flags"] and "FLE_MOE_CACHE_BUDGET_MIB" in table["env"]
    status, out = api.call(
        "POST", "/api/flags/explain", {"args": ["-t", "6", "--nonsense"], "env": {"CUDA_VISIBLE_DEVICES": "0"}}
    )
    assert status == 200
    by_name = {i["name"]: i for i in out["items"]}
    assert by_name["-t"]["known"] and by_name["-t"]["value"] == "6"
    assert by_name["--nonsense"]["known"] is False and by_name["--nonsense"]["text"] == ""
    assert by_name["CUDA_VISIBLE_DEVICES"]["known"]


def test_the_cache_view_says_what_is_missing_before_any_data(api: Client) -> None:
    status, snap = api.call("GET", "/api/cache")
    assert status == 200 and snap["available"] == {"overall": False, "layers": False, "hot": False}
    assert len(snap["missing"]) == 3
    api.studio.cache.feed("moe-cache: steps=512 hits=3 misses=1 hit-rate=75.0%")
    assert api.call("GET", "/api/cache")[1]["overall"]["hit_rate"] == 75.0


def test_state_endpoints_start_empty(api: Client) -> None:
    status, tuning = api.call("GET", "/api/tuning")
    assert status == 200 and tuning["current"] is None and tuning["results"] == []
    assert {p["key"] for p in tuning["params"]} >= {"threads", "cache", "ncmoe", "batch"}
    status, qual = api.call("GET", "/api/quality")
    assert status == 200 and qual["results"] == [] and qual["builtin"]["license"] == "CC0-1.0"
    assert api.call("GET", "/api/tuning/best?model=abc")[1] == {"profile": None}


def test_mistakes_are_answered_with_a_message(api: Client) -> None:
    status, err = api.call("POST", "/api/tuning/start", {})
    assert status == 400 and "model" in err["error"]["message"].lower()
    status, err = api.call("POST", "/api/quality/probe", {})
    assert status == 400 and "Load a model" in err["error"]["message"]
    status, err = api.call("POST", "/api/tuning/aba", {"model": "x"})
    assert status == 400
    assert api.call("POST", "/api/quality/cancel")[0] == 200
    assert api.call("POST", "/api/tuning/cancel")[0] == 200
    assert api.call("POST", "/api/tuning/abc123/delete")[0] == 200


def test_the_speed_test_refuses_a_missing_question_set(api: Client) -> None:
    status, err = api.call("POST", "/api/bench/start", {"runs": 1, "knowledge": True, "knowledge_set": "mkqa"})
    assert status == 400 and "MKQA" in err["error"]["message"]
