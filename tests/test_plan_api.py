"""POST /api/engine/plan: the read-only plan behind the live preview in Engine > Advanced, with its one-click fixes."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fle_studio import catalog
from fle_studio.server import Studio, make_server
from fle_studio.system import GiB

CAT = next(m for m in catalog.CATALOG if m.mtp)
GPU = {"vendor": "nvidia", "index": 0, "name": "RTX", "memory_total": 16 * GiB, "memory_used": 0}
SYSINFO = {
    "cpu": {"physical": 8, "logical": 16, "name": "cpu"},
    "memory": {"total": 64 * GiB},
    "gpus": [GPU],
    "platform": "linux",
}
ENGINE = {"kind": "fle", "flavor": "fle", "path": "/engines/llama-server", "backend": "cuda", "spec_type": True}


def model(*, with_sidecars: bool, catalog_id: str | None = CAT.id) -> dict[str, Any]:
    mtp = {"supported": True, "head": "/m/head.gguf", "vocab": "/m/vocab.txt"} if with_sidecars else {"supported": True}
    return {
        "id": "abc123abc123",
        "name": "M",
        "path": "/models/m.gguf",
        "complete": True,
        "catalog_id": catalog_id,
        "mtp": mtp,
        "info": {
            "context_length": 32768,
            "gpu_bytes": 2 * GiB,
            "expert_bytes": 20 * GiB,
            "kv_bytes_per_token_q8": 1024,
        },
    }


class Client:
    def __init__(self, studio: Studio, port: int) -> None:
        self.studio, self.port = studio, port

    def post(self, path: str, body: dict[str, Any]) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = {"X-FLE-Token": self.studio.token, "Content-Type": "application/json"}
        conn.request("POST", path, body=json.dumps(body), headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"null")
        conn.close()
        return resp.status, data


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Client]:
    st = Studio(tmp_path / "home")
    monkeypatch.setattr(st, "sysinfo", lambda fresh=False: SYSINFO)
    monkeypatch.setattr(st, "engine_for", lambda want, info: dict(ENGINE))
    server = make_server(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield Client(st, int(server.server_address[1]))
    server.shutdown()
    server.server_close()


def use(api: Client, monkeypatch: pytest.MonkeyPatch, m: dict[str, Any] | None, **launch: Any) -> None:
    monkeypatch.setattr(api.studio.library, "get", lambda ident: m)
    api.studio.config.update({"launch": launch})


def test_plan_follows_the_saved_settings(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=True))
    status, out = api.post("/api/engine/plan", {"model": "abc123abc123"})
    assert status == 200 and out["ok"] and out["fixes"] == []
    assert "draft-mtp" in out["plan"]["args"] and "draft-mtp" in out["plan"]["command"]
    api.studio.config.update({"launch": {"spec": "none"}})  # the next preview sees the new setting
    out = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]
    assert out["ok"] and "--spec-type" not in out["plan"]["args"]
    assert any("Speculation is off" in n for n in out["plan"]["notes"])


def test_plan_does_not_start_anything(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=True))
    api.post("/api/engine/plan", {"model": "abc123abc123"})
    assert api.studio.engine.status()["state"] == "stopped"


def test_a_plan_error_is_an_answer_with_a_readable_message(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=True), spec="mtp", mtp=False)
    status, out = api.post("/api/engine/plan", {"model": "abc123abc123"})
    assert status == 200 and out["ok"] is False
    assert "contradict" in out["error"] and "(" not in out["error"][:2]  # text, not a tuple repr
    assert out["fixes"] == []  # downloading would not resolve a contradiction


def test_no_model_is_an_answer_too(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, None)
    status, out = api.post("/api/engine/plan", {"model": ""})
    assert status == 200 and out["ok"] is False and "model" in out["error"].lower()


def test_missing_sidecars_offer_the_download_with_auto(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=False))
    out = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]
    assert out["ok"] and any("MTP is off" in n for n in out["plan"]["notes"])
    assert [f["kind"] for f in out["fixes"]] == ["download_mtp"]
    assert out["fixes"][0]["model"] == CAT.id and out["fixes"][0]["label"] == "Download MTP files"
    assert out["fixes"][0]["size"] == catalog.MTP_HEAD.size + catalog.MTP_VOCAB.size


def test_missing_sidecars_offer_the_download_with_mtp_even_though_the_plan_fails(
    api: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    use(api, monkeypatch, model(with_sidecars=False), spec="mtp")
    out = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]
    assert out["ok"] is False and "sidecar" in out["error"].lower()
    assert out["fixes"] == [{**out["fixes"][0], "kind": "download_mtp", "model": CAT.id}]


@pytest.mark.parametrize(
    "launch",
    [{"spec": "none"}, {"spec": "ngram-mod"}, {"mtp": False}, {"profile": "some-profile"}],
    ids=["spec-none", "spec-ngram", "mtp-switched-off", "profile"],
)
def test_no_download_offer_when_mtp_would_not_be_used(
    api: Client, monkeypatch: pytest.MonkeyPatch, launch: dict[str, Any]
) -> None:
    use(api, monkeypatch, model(with_sidecars=False), **launch)
    assert api.post("/api/engine/plan", {"model": "abc123abc123", "profile": launch.get("profile")})[1]["fixes"] == []


def test_no_download_offer_for_a_model_outside_the_catalog(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=False, catalog_id=None))
    assert api.post("/api/engine/plan", {"model": "abc123abc123"})[1]["fixes"] == []


def test_the_fix_runs_the_download_the_models_page_uses(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        api.studio, "download_catalog", lambda ident, with_mtp: calls.append((ident, with_mtp)) or {"id": "job"}
    )
    use(api, monkeypatch, model(with_sidecars=False))
    fix = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]["fixes"][0]
    status, job = api.post(f"/api/catalog/{fix['model']}/download", {"mtp": True})
    assert status == 200 and job == {"id": "job"} and calls == [(CAT.id, True)]


def test_speculation_applies_with_the_fle_engine(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=True))
    out = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]
    assert out["ok"] and out["speculation"] == {"applies": True, "reason": ""}


def test_speculation_does_not_apply_with_an_upstream_engine(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = {"kind": "upstream", "flavor": "upstream", "path": "/engines/up/llama-server", "backend": "cuda"}
    monkeypatch.setattr(api.studio, "engine_for", lambda want, info: dict(upstream))
    use(api, monkeypatch, model(with_sidecars=True), spec="ngram-mod")
    out = api.post("/api/engine/plan", {"model": "abc123abc123"})[1]
    assert out["ok"] and "--spec-type" not in out["plan"]["args"]  # the command really is unchanged by the setting
    assert out["speculation"]["applies"] is False
    assert "FLE engine" in out["speculation"]["reason"] and "upstream" in out["speculation"]["reason"]


def test_speculation_answer_also_comes_with_a_failed_plan(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = {"kind": "upstream", "flavor": "upstream", "path": "/engines/up/llama-server", "backend": "cuda"}
    monkeypatch.setattr(api.studio, "engine_for", lambda want, info: dict(upstream))
    use(api, monkeypatch, None)  # no model: the plan cannot be built, the engine is still known
    out = api.post("/api/engine/plan", {"model": ""})[1]
    assert out["ok"] is False and out["speculation"]["applies"] is False


def test_speculation_stays_available_when_the_engine_is_unknown(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    def none_installed(want: str, info: Any) -> dict[str, Any]:
        raise RuntimeError("no engine installed")

    monkeypatch.setattr(api.studio, "engine_for", none_installed)
    use(api, monkeypatch, None)
    out = api.post("/api/engine/plan", {"model": ""})[1]
    assert out["ok"] is False and out["speculation"] == {"applies": True, "reason": ""}


def test_speculation_does_not_apply_to_a_profile(api: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    use(api, monkeypatch, model(with_sidecars=True))
    out = api.post("/api/engine/plan", {"model": "abc123abc123", "profile": "some-profile"})[1]
    assert out["speculation"]["applies"] is False and "profile" in out["speculation"]["reason"]
