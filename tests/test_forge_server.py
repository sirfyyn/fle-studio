"""The Forge API end to end: a real Studio server, the stand-in llama tools, tiny synthetic models."""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from forge_synth import make_model

from fle_studio.forge import recipe as recipes
from fle_studio.server import Studio, make_server

FAKES = Path(__file__).parent / "fake_tools"


class Client:
    def __init__(self, studio: Studio, port: int) -> None:
        self.studio, self.port = studio, port

    def call(self, method: str, path: str, body: Any = None, token: bool = True) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-FLE-Token"] = self.studio.token
        conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        return resp.status, json.loads(raw) if raw else None

    def get(self, path: str) -> tuple[int, Any]:
        return self.call("GET", path)

    def post(self, path: str, body: Any = None) -> tuple[int, Any]:
        return self.call("POST", path, {} if body is None else body)

    def wait(self, ident: str, timeout: float = 60) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            _, job = self.get(f"/api/forge/jobs/{ident}")
            if job["state"] in ("done", "failed", "cancelled") and not job["running"]:
                return job
            time.sleep(0.05)
        raise AssertionError("job did not finish")


@pytest.fixture
def api(tmp_path: Path, monkeypatch) -> Iterator[Client]:
    monkeypatch.delenv("FAKE_DELAY", raising=False)
    monkeypatch.delenv("FAKE_FAIL", raising=False)
    st = Studio(tmp_path / "home")
    server = make_server(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield Client(st, int(server.server_address[1]))
    server.shutdown()
    server.server_close()


@pytest.fixture
def models(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "eval.txt").write_text("evaluation text\n" * 40, encoding="utf-8")
    return {
        "high": str(make_model(tmp_path / "high.gguf", seed=1, layers=4, types=[("Q8_0",) * 3] * 4)),
        "low": str(make_model(tmp_path / "low.gguf", seed=2, layers=4, types=[("Q4_0",) * 3] * 4)),
        "eval": str(tmp_path / "eval.txt"),
    }


def test_the_forge_pages_need_the_page_token(api: Client):
    assert api.call("POST", "/api/forge/check", {}, token=False)[0] == 403
    assert api.call("POST", "/api/forge/tools", {"tools_dir": "/x"}, token=False)[0] == 403
    assert api.call("POST", "/api/forge/jobs", {}, token=False)[0] == 403
    assert api.get("/api/forge")[0] == 200


def test_the_overview_lists_tools_defaults_and_types(api: Client):
    status, data = api.post("/api/forge/tools", {"tools_dir": str(FAKES)})
    assert status == 200
    names = ["llama-imatrix", "llama-quantize", "llama-perplexity", "llama-fle-reap"]
    assert [t["name"] for t in data["tools"]] == names
    assert all(t["ok"] for t in data["tools"]) and data["tools_dir"] == str(FAKES)
    assert data["defaults"]["schema"] == 1 and "iq3_s" in data["types"] and "reap" in data["scores"]
    assert data["output_dir"].endswith("forge") and data["references"]
    status, again = api.get("/api/forge")
    assert status == 200 and again["tools_dir"] == str(FAKES)  # the folder is remembered
    status, cleared = api.post("/api/forge/tools", {"tools_dir": ""})
    assert status == 200 and cleared["tools_dir"] == ""


def test_inspect_and_calculate_a_hybrid(api: Client, models):
    status, d = api.post("/api/forge/inspect", {"path": models["low"]})
    assert status == 200 and d["layers"] == 4 and d["experts"] == 8
    assert api.post("/api/forge/inspect", {"path": "/no/such/file.gguf"})[0] == 400
    body = {
        "mode": "hybrid",
        "base": "H",
        "sources": {"H": models["high"], "L": models["low"]},
        "budget_gb": 1,
        "ram_gb": 64,
        "vram_gb": 24,
    }
    status, calc = api.post("/api/forge/calculate", body)
    assert status == 200 and calc["selection"]["feasible"] and set(calc["layers"]) == {"0", "1", "2", "3"}
    assert calc["preview"]["verdict"] == "all_gpu" and calc["file_bytes"] > 0
    assert api.post("/api/forge/calculate", {**body, "budget_gb": 0})[0] == 400
    assert api.post("/api/forge/calculate", {**body, "base": "X"})[0] == 400
    status, fixed = api.post("/api/forge/calculate", {**body, "layers": {"1": "L"}, "reap_keep": 4})
    assert status == 200 and fixed["selection"] is None and fixed["pruned_bytes"] < fixed["file_bytes"]
    status, q = api.post(
        "/api/forge/calculate", {"mode": "quantize", "source_path": models["high"], "quantize": {"type": "q4_0"}}
    )
    assert status == 200 and q["estimate"]["estimate_bytes"] > 0


def test_a_corpus_can_be_mixed_from_files(api: Client, models):
    status, made = api.post("/api/forge/corpus", {"inputs": [models["eval"]], "chars": 2000})
    assert status == 200 and Path(made["path"]).is_file()
    assert api.post("/api/forge/corpus", {"inputs": []})[0] == 400
    status, bundled = api.post("/api/forge/corpus", {"bundled": True})
    assert status == 200 and bundled["part"] == "calibration" and Path(bundled["path"]).is_file()
    status, held = api.post("/api/forge/corpus", {"bundled": True, "part": "evaluation"})
    assert status == 200 and held["path"] != bundled["path"] and held["chars"] < bundled["chars"]


def test_recipes_are_imported_with_what_they_need(api: Client):
    shared = recipes.portable(
        recipes.normalize({"name": "friend", "sources": {"B": {"path": "/x/base.gguf", "role": "base"}}})
    )
    status, data = api.post("/api/forge/recipe/import", {"recipe": shared})
    assert status == 200 and data["needs"] == [
        {"id": "B", "file": "base.gguf", "role": "base", "size": 0, "origin": ""}
    ]
    status, share = api.post(
        "/api/forge/recipe/share", {"recipe": {"name": "mine", "sources": {"B": {"path": "/home/me/b.gguf"}}}}
    )
    assert status == 200 and share["recipe"]["sources"]["B"] == {
        **share["recipe"]["sources"]["B"],
        "path": "",
        "file": "b.gguf",
    }
    assert api.post("/api/forge/recipe/share", {"recipe": {"mode": "magic"}})[0] == 400
    assert api.post("/api/forge/recipe/import", {"recipe": {"mode": "magic"}})[0] == 400
    assert api.post("/api/forge/recipe/import", {"recipe": {"schema": 7}})[0] == 400


def test_a_job_through_the_api_and_publishing_the_result(api: Client, models, monkeypatch):
    monkeypatch.setenv("FAKE_DELAY", "0.2")  # keeps the first job running while the second is refused
    api.post("/api/forge/tools", {"tools_dir": str(FAKES)})
    recipe = {
        "name": "viaapi",
        "mode": "quantize",
        "sources": {"S": {"path": models["high"], "role": "base"}},
        "quantize": {"source": "S", "type": "q4_0", "imatrix": False},
        "verify": {"enabled": True, "reference": "S", "text": models["eval"], "chunks": 2, "context": 128},
    }
    status, info = api.post("/api/forge/check", {"recipe": recipe})
    assert status == 200 and info["ok"] and [s["id"] for s in info["steps"]] == ["build", "kld_base", "kld", "finish"]
    bad = api.post("/api/forge/check", {"recipe": {"mode": "magic"}})
    assert bad[0] == 400
    status, job = api.post("/api/forge/jobs", {"recipe": recipe})
    assert status == 200 and job["state"] in ("queued", "running")
    ident = job["id"]
    status, busy = api.post("/api/forge/jobs", {"recipe": recipe})
    assert status == 409 and "Another Forge job" in busy["error"]["message"]
    done = api.wait(ident)
    assert done["state"] == "done", done["error"]
    status, log = api.get(f"/api/forge/jobs/{ident}/log?since=0")
    assert status == 200 and log["lines"] and log["next"] >= len(log["lines"])
    status, tail = api.get(f"/api/forge/jobs/{ident}/log?since={log['next']}")
    assert tail["lines"] == []
    status, shared = api.get(f"/api/forge/jobs/{ident}/recipe")
    assert status == 200 and shared["sources"]["S"]["path"] == "" and shared["name"] == "viaapi"
    status, listing = api.get("/api/forge")
    assert any(j["id"] == ident for j in listing["jobs"])

    status, pub = api.post(f"/api/forge/jobs/{ident}/publish", {})
    assert status == 200 and pub["profile"]["name"] == "Forge: viaapi"
    assert pub["model"]
    assert api.post("/api/forge/jobs/" + "0" * 12 + "/cancel", {})[0] == 404
    assert api.get("/api/forge/jobs/" + "0" * 12)[0] == 404
    assert api.post(f"/api/forge/jobs/{ident}/resume", {})[0] == 409  # a finished job cannot be resumed
    assert api.post(f"/api/forge/jobs/{ident}/delete", {})[0] == 200
    assert api.get(f"/api/forge/jobs/{ident}")[0] == 404


def test_a_job_cannot_be_published_before_it_is_done(api: Client, models):
    api.post("/api/forge/tools", {"tools_dir": str(FAKES)})
    recipe = {
        "name": "early",
        "mode": "hybrid",
        "verify": {"enabled": False},
        "sources": {"B": {"path": models["high"], "role": "base"}, "L": {"path": models["low"]}},
        "hybrid": {"base": "B", "layers": {}},
    }
    status, job = api.post("/api/forge/jobs", {"recipe": recipe})
    assert status == 200
    done = api.wait(job["id"])
    assert done["state"] == "failed"  # nothing taken from a donor
    assert api.post(f"/api/forge/jobs/{job['id']}/publish", {})[0] == 409
