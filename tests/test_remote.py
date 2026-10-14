"""Host and client mode with two real Studios in one process: pairing, rights, the pass-through (streaming included),
revoking, a changed certificate, mirrored chats and the SSH tunnel (with a stand-in for ssh that really forwards)."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import ssl
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from e2e.run_e2e import gguf_stub
from test_service import server_unit, settle, systemd  # noqa: F401 - systemd is a fixture

from fle_studio import remote, tlsgen
from fle_studio.server import Studio, make_server

HERE = Path(__file__).parent
FAKE_ENGINE = HERE / "fake_llama_server.py"
FAKE_SSH = HERE / "fake_ssh.py"
NO_VERIFY = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
NO_VERIFY.check_hostname = False
NO_VERIFY.verify_mode = ssl.CERT_NONE


def wait_for(probe: Callable[[], Any], what: str, timeout: float = 30.0) -> Any:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = probe()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class Web:
    """One side of the wire: a Studio's address, with or without its page token or a client token."""

    def __init__(self, port: int, tls: bool = False, page: str = "", bearer: str = "") -> None:
        self.port, self.tls, self.page, self.bearer = port, tls, page, bearer

    def conn(self, timeout: float = 30) -> http.client.HTTPConnection:
        if self.tls:
            return http.client.HTTPSConnection("127.0.0.1", self.port, timeout=timeout, context=NO_VERIFY)
        return http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)

    def headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.page:
            h["X-FLE-Token"] = self.page
        if self.bearer:
            h["Authorization"] = "Bearer " + self.bearer
        return {**h, **(extra or {})}

    def call(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
        c = self.conn()
        try:
            data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
            c.request(method, path, body=data, headers=self.headers(headers))
            r = c.getresponse()
            raw = r.read()
        finally:
            c.close()
        try:
            return r.status, json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            return r.status, raw

    def stream(self, path: str, body: Any, headers: dict[str, str] | None = None) -> tuple[int, list[bytes]]:
        c = self.conn(60)
        try:
            c.request("POST", path, body=json.dumps(body).encode(), headers=self.headers(headers))
            r = c.getresponse()
            chunks = []
            while True:
                piece = r.read1(65536)
                if not piece:
                    break
                chunks.append(piece)
        finally:
            c.close()
        return r.status, chunks


class Rig:
    """A Studio with its server on a loopback port."""

    def __init__(self, home: Path, tls: bool) -> None:
        self.studio = Studio(home)
        self.server = make_server(self.studio, "127.0.0.1", 0, tls=tls)
        self.port = int(self.server.server_address[1])
        self.tls = tls
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.owner = Web(self.port, tls, page=self.studio.token)  # the person at that machine
        self.anonymous = Web(self.port, tls)
        self.alive = True

    @property
    def fingerprint(self) -> str:
        return self.studio.tls_fingerprint

    def stop(self) -> None:
        if self.alive:
            self.alive = False
            self.server.shutdown()
            self.server.server_close()
        remote.shutdown(self.studio)

    def code(self) -> str:
        status, data = self.owner.call("POST", "/api/remote/code", {})
        assert status == 200, data
        return str(data["code"])

    def pair(self, name: str = "test client", admin: bool = False) -> Web:
        """Direct pairing, like the client Studio does it; returns a Web that carries the new client token."""
        proof = remote.pair_proof(self.code(), self.fingerprint)
        status, data = self.anonymous.call("POST", "/api/remote/pair", {"proof": proof, "name": name})
        assert status == 200, data
        client = Web(self.port, self.tls, bearer=data["token"])
        if admin:
            assert (
                self.owner.call("POST", f"/api/remote/clients/{data['client']['id']}/admin", {"admin": True})[0] == 200
            )
        client.client_id = data["client"]["id"]  # type: ignore[attr-defined]
        return client


@pytest.fixture
def rigs(tmp_path: Path) -> Iterator[Callable[..., Rig]]:
    made: list[Rig] = []

    def make(name: str, tls: bool) -> Rig:
        rig = Rig(tmp_path / name, tls)
        made.append(rig)
        return rig

    yield make
    for rig in made:
        rig.stop()


def start_fake_engine(rig: Rig, tmp_path: Path) -> None:
    """The stand-in llama-server, loaded on that Studio like a person would do it."""
    stub = tmp_path / f"stub-{rig.port}" / "e2e-stub.gguf"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(gguf_stub())
    web = rig.owner
    assert web.call("POST", "/api/engines/check", {"path": str(FAKE_ENGINE)})[1]["ok"]
    assert (
        web.call("POST", "/api/settings", {"engine": {"choice": "custom", "custom_path": str(FAKE_ENGINE)}})[0] == 200
    )
    model = web.call("POST", "/api/models/add", {"path": str(stub)})[1]
    assert web.call("POST", "/api/engine/start", {"model": model["id"], "profile": "auto"})[0] == 200
    wait_for(lambda: web.call("GET", "/api/engine")[1]["state"] == "ready", "the engine", 60)


@pytest.fixture
def engine_host(rigs: Callable[..., Rig], tmp_path: Path) -> Iterator[Rig]:
    rig = rigs("host", True)
    start_fake_engine(rig, tmp_path)
    yield rig
    rig.owner.call("POST", "/api/engine/stop", {})


def add_host(client: Rig, host: Rig, **extra: Any) -> dict[str, Any]:
    body = {"kind": "direct", "address": f"127.0.0.1:{host.port}", "code": host.code(), **extra}
    status, data = client.owner.call("POST", "/api/hosts", body)
    assert status == 200, data
    return data


# ------------------------------------------------------------------------------------------------ certificate


def test_the_certificate_is_a_real_one_and_the_fingerprint_matches(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    c = host.owner.conn()
    c.connect()
    der = c.sock.getpeercert(binary_form=True)  # type: ignore[union-attr]
    c.close()
    assert hashlib.sha256(der).hexdigest() == host.fingerprint
    cert, key, again = tlsgen.ensure(host.studio.data_dir)
    assert again == host.fingerprint  # the same files again, not a new certificate on every start
    # Does a stricter client accept it when the certificate is trusted explicitly (a valid signature, a usable key)?
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(str(cert))
    ctx.check_hostname = False
    c = http.client.HTTPSConnection("127.0.0.1", host.port, context=ctx, timeout=10)
    c.request("GET", "/api/ping")
    assert c.getresponse().status == 200
    c.close()
    if os.name != "nt":
        assert (key.stat().st_mode & 0o077) == 0  # the private key is readable by the owner only


def test_host_mode_without_tls_and_loopback_servers_stay_plain(rigs: Callable[..., Rig]) -> None:
    plain = rigs("plain", False)
    assert plain.studio.tls_fingerprint == "" and plain.studio.listen["tls"] is False
    assert plain.owner.call("GET", "/api/ping")[0] == 200


# ------------------------------------------------------------------------------------------------ pairing


def test_pairing_gives_a_token_once_and_never_logs_or_stores_it(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    assert host.anonymous.call("POST", "/api/remote/pair", {"proof": "x", "name": "n"})[1]["error"]["code"] == "no_code"
    client = host.pair("Max's Mac")
    token = client.bearer
    assert token.startswith("fcl_") and len(token) > 40
    status, who = client.call("GET", "/api/remote/whoami")
    assert status == 200 and who["paired"] and who["name"] == "Max's Mac" and not who["admin"]
    listing = host.owner.call("GET", "/api/remote")[1]
    assert [c["name"] for c in listing["clients"]] == ["Max's Mac"] and "hash" not in listing["clients"][0]
    assert token not in json.dumps(listing)
    stored = (host.studio.data_dir / "clients.json").read_text(encoding="utf-8")
    assert token not in stored and remote.token_hash(token) in stored  # only a hash is kept
    # the code is single use
    proof = remote.pair_proof("ABCDEFGH", host.fingerprint)
    assert (
        host.anonymous.call("POST", "/api/remote/pair", {"proof": proof, "name": "again"})[1]["error"]["code"]
        == "no_code"
    )


def test_the_proof_is_bound_to_the_certificate_the_client_saw(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    code = host.code()
    wrong = remote.pair_proof(code, "ab" * 32)  # a client that saw some other certificate (a man in the middle)
    status, body = host.anonymous.call("POST", "/api/remote/pair", {"proof": wrong, "name": "x"})
    assert status == 400 and body["error"]["code"] == "bad_code"
    assert code not in json.dumps(body)
    ok = remote.pair_proof(code, tlsgen.pretty(host.fingerprint))  # the formatting of the fingerprint does not matter
    assert host.anonymous.call("POST", "/api/remote/pair", {"proof": ok, "name": "x"})[0] == 200


def test_code_expires_and_wrong_tries_are_limited(rigs: Callable[..., Rig], monkeypatch: pytest.MonkeyPatch) -> None:
    host = rigs("host", True)
    code = host.code()
    assert host.owner.call("GET", "/api/remote")[1]["code"]["active"]
    for _ in range(remote.FAILS_PER_ADDRESS):
        assert host.anonymous.call("POST", "/api/remote/pair", {"proof": "no", "name": "x"})[0] == 400
    status, body = host.anonymous.call("POST", "/api/remote/pair", {"proof": remote.pair_proof(code, host.fingerprint)})
    assert status == 429 and body["error"]["code"] == "rate_limited"  # even the right proof waits
    later = time.time() + remote.FAIL_WINDOW + 5
    monkeypatch.setattr(remote, "_now", lambda: later)
    assert (
        host.anonymous.call("POST", "/api/remote/pair", {"proof": remote.pair_proof(code, host.fingerprint)})[1][
            "error"
        ]["code"]
        == "no_code"
    )  # the limit is over, but the code ran out after ten minutes


def test_a_guessed_at_code_is_thrown_away(tmp_path: Path) -> None:
    clients = remote.Clients(tmp_path)
    code = clients.new_code()["code"]
    for i in range(remote.CODE_BURNED_AFTER):
        with pytest.raises(remote.RemoteError):
            clients.pair("wrong", "x", f"10.0.0.{i}", "")  # many addresses: only the count of the code matters
    with pytest.raises(remote.RemoteError) as gone:
        clients.pair(remote.pair_proof(code, ""), "x", "10.0.0.99", "")
    assert gone.value.code == "no_code"
    assert not clients.code_state()["active"]


def test_codes_are_short_and_readable() -> None:
    code = remote.Clients(Path("/nonexistent-never-written")).code_state()
    assert code == {"active": False, "expires": 0}
    assert len(remote.CODE_ALPHABET) == 32 and not set("01OI") & set(remote.CODE_ALPHABET)
    assert remote.normalize_code("abcd-efgh ") == "ABCDEFGH"
    assert remote.pair_proof("abcd-efgh", "AA:BB") == remote.pair_proof("ABCDEFGH", "aabb")


# ------------------------------------------------------------------------------------------------ rights


def test_standard_clients_chat_look_and_measure_but_change_nothing(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    c = host.pair()
    assert c.call("GET", "/api/state")[0] == 200  # look
    assert c.call("GET", "/api/settings")[0] == 200 and "hf_token" not in c.call("GET", "/api/settings")[1]
    status, chat = c.call(
        "POST", "/api/chats", {"messages": [{"role": "user", "content": "hi"}]}
    )  # chat, no page token
    assert status == 200 and c.call("GET", f"/api/chats/{chat['id']}")[0] == 200
    assert c.call("POST", "/api/bench/cancel", {})[0] == 200  # measure
    for method, path in [
        ("POST", "/api/settings"),
        ("POST", "/api/engines/install"),
        ("POST", "/api/engine/stop"),
        ("POST", "/api/models/add"),
        ("POST", "/api/engine/service/start"),
        ("POST", "/api/profiles"),
        ("POST", "/api/hf/download"),
        ("DELETE", "/api/anything-new"),
    ]:
        status, body = c.call(method, path, {})
        assert status == 403 and body["error"]["code"] == "no_admin", (path, status, body)


def test_admin_clients_may_change_things_and_installs_ask_for_a_confirmation(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    c = host.pair(admin=True)
    assert c.call("POST", "/api/settings", {"theme": "dark"})[0] == 200
    status, body = c.call("POST", "/api/models/0123456789ab/remove", {})
    assert status == 428 and body["error"]["confirm"] is True
    status, _ = c.call("POST", "/api/models/0123456789ab/remove", {}, headers={"X-FLE-Confirm": "1"})
    assert status != 428
    for path in ("/api/engines/install", "/api/engines/some-id/remove", "/api/update/apply"):
        assert c.call("POST", path, {})[0] == 428, path
    # taking the right away takes effect at once
    assert host.owner.call("POST", f"/api/remote/clients/{c.client_id}/admin", {"admin": False})[0] == 200  # type: ignore[attr-defined]
    assert c.call("POST", "/api/settings", {"theme": "light"})[0] == 403


def test_a_client_never_manages_clients_or_hosts_not_even_as_admin(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    c = host.pair(admin=True)
    for method, path in [
        ("GET", "/api/remote"),
        ("POST", "/api/remote/code"),
        ("GET", "/api/hosts"),
        ("POST", "/api/hosts"),
        ("POST", f"/api/remote/clients/{c.client_id}/revoke"),  # type: ignore[attr-defined]
        ("POST", "/h/0123456789ab/api/state"),
        ("GET", "/h/0123456789ab/api/state"),
    ]:
        status, body = c.call(method, path, {} if method == "POST" else None)
        assert status == 403 and body["error"]["code"] == "owner_only", (path, status, body)


def test_revoking_ends_access_at_once(rigs: Callable[..., Rig]) -> None:
    host = rigs("host", True)
    c = host.pair()
    assert c.call("GET", "/api/state")[0] == 200
    assert host.owner.call("POST", f"/api/remote/clients/{c.client_id}/revoke", {})[0] == 200  # type: ignore[attr-defined]
    assert c.call("GET", "/api/state")[0] == 401
    assert host.owner.call("GET", "/api/remote")[1]["clients"] == []
    assert host.owner.call("POST", "/api/remote/clients/0123456789ab/revoke", {})[0] == 404


def test_the_browser_way_is_unchanged(rigs: Callable[..., Rig]) -> None:
    """Loopback needs no token to look; changes still need the page token. A bogus client token is refused."""
    host = rigs("host", True)
    assert host.anonymous.call("GET", "/api/state")[0] == 200
    assert host.anonymous.call("POST", "/api/settings", {"theme": "dark"})[0] == 403  # no page token
    assert host.owner.call("POST", "/api/settings", {"theme": "dark"})[0] == 200
    assert Web(host.port, True, bearer="fcl_not-a-token").call("GET", "/api/state")[0] == 401


# ------------------------------------------------------------------------------------------------ the client Studio


def test_add_a_host_and_use_every_page_through_the_pass_through(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    added = add_host(client, engine_host)
    host = added["host"]
    assert host["fingerprint"] == engine_host.fingerprint and host["kind"] == "direct" and host["color"]
    assert added["probe"]["ok"] and added["probe"]["host_name"]
    stored = (client.studio.data_dir / "hosts.json").read_text(encoding="utf-8")
    token = json.loads(stored)[0]["token"]
    assert token.startswith("fcl_") and token not in json.dumps(client.owner.call("GET", "/api/hosts")[1])
    base = f"/h/{host['id']}"
    me = client.owner
    # state and live data are the host's
    assert me.call("GET", f"{base}/api/state")[1]["data_dir"] == str(engine_host.studio.data_dir)
    assert me.call("GET", f"{base}/api/engine")[1]["state"] == "ready"
    assert me.call("GET", "/api/engine")[1]["state"] == "stopped"  # this machine is not
    # chats are made on the host and not here
    status, chat = me.call("POST", f"{base}/api/chats", {"messages": [{"role": "user", "content": "from the client"}]})
    assert status == 200
    assert [c["id"] for c in me.call("GET", f"{base}/api/chats")[1]["chats"]] == [chat["id"]]
    assert me.call("GET", "/api/chats")[1]["chats"] == []
    assert engine_host.owner.call("GET", f"/api/chats/{chat['id']}")[1]["messages"][0]["content"] == "from the client"
    # the page token of this Studio is still needed to change anything
    assert Web(client.port).call("POST", f"{base}/api/chats", {"messages": []})[0] == 403


def test_streaming_chat_runs_on_the_host(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    body = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "Hello through the wire"}]}
    status, chunks = client.owner.stream(f"/h/{host['id']}/v1/chat/completions", body, {"Accept": "text/event-stream"})
    text = b"".join(chunks).decode("utf-8")
    assert status == 200 and len(chunks) > 3 and text.rstrip().endswith("data: [DONE]")
    assert "Fake reply to" in text.replace("\n", "") or '"content"' in text
    # a stream arrives while it is made: more than one piece, not one block at the end
    assert text.count("data: ") > 10
    # not streamed
    status, data = client.owner.call(
        "POST",
        f"/h/{host['id']}/v1/chat/completions",
        {"model": "x", "messages": [{"role": "user", "content": "plain"}]},
    )
    assert status == 200 and data["choices"][0]["message"]["content"]


def test_large_uploads_and_downloads_pass_through(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    big = "x" * 3_000_000  # an attachment-sized message
    status, chat = client.owner.call(
        "POST", f"/h/{host['id']}/api/chats", {"messages": [{"role": "user", "content": big}]}
    )
    assert status == 200
    status, back = client.owner.call("GET", f"/h/{host['id']}/api/chats/{chat['id']}")
    assert status == 200 and back["messages"][0]["content"] == big
    # the connection is still in order after a large body
    assert client.owner.call("GET", f"/h/{host['id']}/api/ping")[0] == 200


def test_rights_and_confirmations_hold_through_the_pass_through(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    added = add_host(client, engine_host)["host"]
    base = f"/h/{added['id']}"
    status, body = client.owner.call("POST", f"{base}/api/settings", {"theme": "dark"})
    assert status == 403 and body["error"]["code"] == "no_admin"
    listing = engine_host.owner.call("GET", "/api/remote")[1]
    engine_host.owner.call("POST", f"/api/remote/clients/{listing['clients'][0]['id']}/admin", {"admin": True})
    assert client.owner.call("POST", f"{base}/api/settings", {"theme": "dark"})[0] == 200
    status, body = client.owner.call("POST", f"{base}/api/models/0123456789ab/remove", {})
    assert status == 428 and body["error"]["confirm"] is True  # the page asks the person, then sends the header
    status, _ = client.owner.call("POST", f"{base}/api/models/0123456789ab/remove", {}, headers={"X-FLE-Confirm": "1"})
    assert status != 428
    # nothing that manages hosts or clients is passed on, and the token header of the page is not forwarded
    for path in ("/api/remote/code", "/api/remote", "/api/hosts", "/other"):
        assert client.owner.call("GET", f"{base}{path}")[0] == 403, path


def test_a_revoked_client_is_told_to_pair_again_and_can(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    clients = engine_host.owner.call("GET", "/api/remote")[1]["clients"]
    engine_host.owner.call("POST", f"/api/remote/clients/{clients[0]['id']}/revoke", {})
    status, body = client.owner.call("GET", f"/h/{host['id']}/api/state")
    assert status == 424 and body["error"]["code"] == "unpaired"  # not 401: the page would send you to the login
    check = client.owner.call("POST", f"/api/hosts/{host['id']}/check", {})[1]
    assert check["ok"] is False and check["code"] == "unpaired"
    status, data = client.owner.call("POST", f"/api/hosts/{host['id']}/repair", {"code": engine_host.code()})
    assert status == 200 and data["probe"]["ok"]
    assert client.owner.call("GET", f"/h/{host['id']}/api/state")[0] == 200
    assert client.owner.call("POST", f"/api/hosts/{host['id']}/repair", {"code": "WRONGCODE"})[0] == 400


def test_a_changed_certificate_is_refused_before_anything_is_sent(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    hosts_file = client.studio.data_dir / "hosts.json"
    record = json.loads(hosts_file.read_text(encoding="utf-8"))
    record[0]["fingerprint"] = "ab" * 32  # what was pinned is not what the host shows now
    hosts_file.write_text(json.dumps(record), encoding="utf-8")
    before = engine_host.owner.call("GET", "/api/remote")[1]["clients"][0]["last_seen"]
    status, body = client.owner.call("GET", f"/h/{host['id']}/api/state")
    err = body["error"]
    assert status == 502 and err["code"] == "fingerprint_changed"
    assert err["seen"] == tlsgen.pretty(engine_host.fingerprint) and err["pinned"] == tlsgen.pretty("ab" * 32)
    check = client.owner.call("POST", f"/api/hosts/{host['id']}/check", {})[1]
    assert check["ok"] is False and check["code"] == "fingerprint_changed"
    assert (
        engine_host.owner.call("GET", "/api/remote")[1]["clients"][0]["last_seen"] == before
    )  # the token never went out
    # a person who trusts the new certificate pairs again; with the fingerprint typed in it is checked as well
    status, _ = client.owner.call(
        "POST", f"/api/hosts/{host['id']}/repair", {"code": engine_host.code(), "fingerprint": "cd" * 32}
    )
    assert status == 400
    status, data = client.owner.call(
        "POST",
        f"/api/hosts/{host['id']}/repair",
        {"code": engine_host.code(), "fingerprint": tlsgen.pretty(engine_host.fingerprint)},
    )
    assert status == 200 and data["host"]["fingerprint"] == engine_host.fingerprint


def test_a_fingerprint_that_was_typed_in_must_match_when_adding(rigs: Callable[..., Rig]) -> None:
    host, client = rigs("host", True), rigs("client", False)
    status, body = client.owner.call(
        "POST",
        "/api/hosts",
        {"kind": "direct", "address": f"127.0.0.1:{host.port}", "code": host.code(), "fingerprint": "ab" * 32},
    )
    assert status == 400 and body["error"]["code"] == "fingerprint_mismatch"
    assert client.owner.call("GET", "/api/hosts")[1]["hosts"] == []
    assert add_host(client, host, fingerprint=tlsgen.pretty(host.fingerprint))["host"]["fingerprint"]


def test_adding_checks_the_input(rigs: Callable[..., Rig]) -> None:
    host, client = rigs("host", True), rigs("client", False)
    post = client.owner.call
    assert post("POST", "/api/hosts", {"kind": "direct", "address": "", "code": "ABCDEFGH"})[0] == 400
    assert post("POST", "/api/hosts", {"kind": "direct", "address": "x", "code": ""})[0] == 400
    assert post("POST", "/api/hosts", {"kind": "other", "address": "x", "code": "ABCDEFGH"})[0] == 400
    status, body = post(
        "POST", "/api/hosts", {"kind": "direct", "address": f"127.0.0.1:{host.port}", "code": "ABCDEFGH"}
    )
    assert status == 400 and body["error"]["code"] == "no_code"  # the host has not made a code
    dead = post("POST", "/api/hosts", {"kind": "direct", "address": "127.0.0.1:9", "code": "ABCDEFGH"})
    assert dead[0] == 502 and dead[1]["error"]["code"] == "unreachable"
    add_host(client, host)
    assert (
        post("POST", "/api/hosts", {"kind": "direct", "address": f"127.0.0.1:{host.port}", "code": host.code()})[0]
        == 409
    )


def test_unreachable_mirrored_chats_and_removing(rigs: Callable[..., Rig]) -> None:
    host, client = rigs("host", True), rigs("client", False)
    status, chat = host.owner.call(
        "POST", "/api/chats", {"title": "On the host", "messages": [{"role": "user", "content": "kept"}]}
    )
    entry = add_host(client, host, mirror=True)["host"]
    base = f"/h/{entry['id']}"
    assert client.owner.call("GET", f"{base}/api/chats")[1]["chats"][0]["title"] == "On the host"
    assert client.owner.call("GET", f"{base}/api/chats/{chat['id']}")[0] == 200  # these two are copied now
    host.stop()
    status, body = client.owner.call("GET", f"{base}/api/state")
    assert status == 502 and body["error"]["code"] == "unreachable"
    # offline reading of the mirror, read only
    status, listing = client.owner.call("GET", f"{base}/api/chats")
    assert status == 200 and listing["chats"][0]["title"] == "On the host" and listing["mirror"] is True
    status, one = client.owner.call("GET", f"{base}/api/chats/{chat['id']}")
    assert status == 200 and one["messages"][0]["content"] == "kept" and one["mirror"] is True
    assert client.owner.call("POST", f"{base}/api/chats", {"messages": []})[0] == 502
    assert client.owner.call("GET", f"{base}/api/chats/ffffffffffffffff")[0] == 502  # never copied
    check = client.owner.call("POST", f"/api/hosts/{entry['id']}/check", {})[1]
    assert check["ok"] is False and check["code"] == "unreachable"
    # without the switch there is no copy
    other = client.owner.call(
        "POST", f"/api/hosts/{entry['id']}/update", {"mirror": False, "name": "GPU box", "color": "#112233"}
    )
    assert other[1]["host"]["name"] == "GPU box" and other[1]["host"]["color"] == "#112233"
    assert client.owner.call("GET", f"{base}/api/chats")[0] == 502  # no switch, no offline copy served
    removed = client.owner.call("POST", f"/api/hosts/{entry['id']}/remove", {})
    assert removed[1]["hosts"] == [] and not (client.studio.data_dir / "mirror" / entry["id"]).exists()
    assert client.owner.call("GET", f"{base}/api/state")[0] == 404


def test_service_mode_goes_along(rigs: Callable[..., Rig], systemd: Any, tmp_path: Path) -> None:  # noqa: F811
    """A Studio connected to a systemd unit (--service) is a host like any other: clients chat on that engine, and
    starting or stopping the unit is an admin right."""
    host = rigs("host", True)
    server_unit(systemd, tmp_path)
    host.studio.engine.attach("", "", "fake")
    standard, admin = host.pair("standard"), host.pair("admin", admin=True)
    assert standard.call("GET", "/api/engine")[1]["service"]["unit"] == "fake.service"
    assert standard.call("POST", "/api/engine/service/start", {})[0] == 403
    assert admin.call("POST", "/api/engine/service/start", {})[0] == 200
    settle(host.studio.engine, "attached")
    body = {"model": "fake", "messages": [{"role": "user", "content": "hello service"}]}
    status, data = standard.call("POST", "/v1/chat/completions", body)
    assert status == 200 and data["choices"][0]["message"]["content"]
    status, chunks = standard.stream("/v1/chat/completions", {**body, "stream": True})
    assert status == 200 and b"[DONE]" in b"".join(chunks)
    assert standard.call("POST", "/api/engine/service/stop", {})[0] == 403
    assert admin.call("POST", "/api/engine/service/stop", {})[0] == 200


# ------------------------------------------------------------------------------------------------ ssh


def test_the_ssh_command_is_built_for_keys_only_and_a_watched_tunnel() -> None:
    cmd = remote.ssh_command(["ssh"], "max@gpu-box", 2222, 51000, 8642)
    assert cmd[:4] == ["ssh", "-N", "-T", "-L"] and cmd[4] == "127.0.0.1:51000:127.0.0.1:8642"
    assert cmd[cmd.index("-p") + 1] == "2222" and cmd[-1] == "max@gpu-box"
    options = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-o"]
    assert "BatchMode=yes" in options and "ExitOnForwardFailure=yes" in options  # no passwords, no half tunnels
    assert any(o.startswith("ServerAliveInterval=") for o in options)
    assert not any("password" in a.lower() for a in cmd)


@pytest.mark.parametrize("target", ["", "-oProxyCommand=evil", "a b", "x;rm -rf /", "$(id)", "user@host name"])
def test_ssh_targets_that_could_be_options_or_commands_are_refused(target: str) -> None:
    with pytest.raises(remote.RemoteError):
        remote.check_ssh(target, 22, 8642)


def test_ssh_target_and_ports() -> None:
    assert remote.check_ssh("max@gpu-box.local", "", "") == ("max@gpu-box.local", 22, 8642)
    assert remote.check_ssh("gpu-box", "2200", "9000") == ("gpu-box", 2200, 9000)
    for bad in (("gpu-box", 0, 1), ("gpu-box", 22, 70000), ("gpu-box", "x", 1)):
        with pytest.raises(remote.RemoteError):
            remote.check_ssh(*bad)


def test_the_ssh_program_can_be_replaced_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLE_STUDIO_SSH", json.dumps(["py", "x.py"]))
    assert remote.ssh_program() == ["py", "x.py"]
    monkeypatch.setenv("FLE_STUDIO_SSH", "/opt/ssh")
    assert remote.ssh_program() == ["/opt/ssh"]
    monkeypatch.delenv("FLE_STUDIO_SSH")
    monkeypatch.setattr(remote.shutil, "which", lambda name: None)
    with pytest.raises(remote.RemoteError, match="ssh"):
        remote.ssh_program()


@pytest.fixture
def fake_ssh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "ssh-calls.jsonl"
    monkeypatch.setenv("FLE_STUDIO_SSH", json.dumps([sys.executable, str(FAKE_SSH)]))
    monkeypatch.setenv("FAKE_SSH_LOG", str(log))
    monkeypatch.delenv("FAKE_SSH_FAIL", raising=False)
    return log


def test_an_ssh_host_through_a_tunnel_that_comes_back(rigs: Callable[..., Rig], fake_ssh: Path) -> None:
    """The host here speaks plain http on loopback, as it does when it is only reachable by ssh."""
    solo = rigs("solo", False)
    client = rigs("client", False)
    body = {
        "kind": "ssh",
        "target": "max@gpu-box",
        "port": 22,
        "remote_port": solo.port,
        "code": solo.code(),
        "name": "GPU box by ssh",
    }
    status, added = client.owner.call("POST", "/api/hosts", body)
    assert status == 200, added
    entry = added["host"]
    assert entry["kind"] == "ssh" and entry["tunnel"]["state"] == "up" and entry["fingerprint"] == ""
    argv = json.loads(fake_ssh.read_text(encoding="utf-8").splitlines()[0])
    assert "max@gpu-box" in argv and "BatchMode=yes" in argv and f"127.0.0.1:{solo.port}" in argv[argv.index("-L") + 1]
    base = f"/h/{entry['id']}"
    assert client.owner.call("GET", f"{base}/api/state")[1]["data_dir"] == str(solo.studio.data_dir)
    assert solo.owner.call("GET", "/api/remote")[1]["clients"][0]["name"]  # paired over the tunnel

    # the ssh process dies: the next request needs a moment, then it works again through a new one
    tunnel = client.studio.__dict__["_remote"].hosts._tunnels[entry["id"]]
    first_port = tunnel.local_port
    tunnel._proc.kill()
    wait_for(lambda: tunnel.state != "up" or tunnel.local_port != first_port, "the tunnel to notice")
    assert wait_for(lambda: tunnel.state == "up", "the tunnel to come back", 30)
    assert wait_for(lambda: client.owner.call("GET", f"{base}/api/state")[0] == 200, "a request through the new tunnel")
    assert len(fake_ssh.read_text(encoding="utf-8").splitlines()) >= 2  # a second ssh was started

    # Reconnect: at once, without the pause
    status, data = client.owner.call("POST", f"/api/hosts/{entry['id']}/connect", {})
    assert status == 200 and data["ok"] and data["host"]["tunnel"]["state"] == "up"
    client.owner.call("POST", f"/api/hosts/{entry['id']}/remove", {})
    assert tunnel.state == "stopped"  # removing the host ends its ssh


def test_a_refused_ssh_login_says_why_and_a_dead_tunnel_ends_with_the_studio(
    rigs: Callable[..., Rig], fake_ssh: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solo, client = rigs("solo", False), rigs("client", False)
    monkeypatch.setenv("FAKE_SSH_FAIL", "max@gpu-box: Permission denied (publickey).")
    status, body = client.owner.call(
        "POST", "/api/hosts", {"kind": "ssh", "target": "max@gpu-box", "remote_port": solo.port, "code": solo.code()}
    )
    assert status == 502 and "Permission denied" in body["error"]["message"]
    assert client.owner.call("GET", "/api/hosts")[1]["hosts"] == []
    monkeypatch.delenv("FAKE_SSH_FAIL")
    wait_for(lambda: True, "nothing")
    ok = client.owner.call(
        "POST", "/api/hosts", {"kind": "ssh", "target": "max@gpu-box", "remote_port": solo.port, "code": solo.code()}
    )
    assert ok[0] == 200
    tunnel = next(iter(client.studio.__dict__["_remote"].hosts._tunnels.values()))
    proc = tunnel._proc
    client.stop()  # what __main__ does when it ends
    assert proc.poll() is not None


def test_the_tunnel_class_alone(fake_ssh: Path, rigs: Callable[..., Rig]) -> None:
    solo = rigs("solo", False)
    t = remote.Tunnel("max@gpu-box", 22, solo.port)
    assert t.state == "stopped" and t.info()["local_port"] == 0
    assert t.wait_ready(20)
    assert Web(t.local_port).call("GET", "/api/ping")[0] == 200
    t.stop()
    assert t.state == "stopped"


WRITE_ROUTES_OF_OTHER_PARTS = [
    # (method, path, what a default client gets, what an admin client gets)
    ("POST", "/api/tools/call", 403, None),  # runs tools on the host: admin only
    ("POST", "/api/tools/reload", 403, None),
    ("POST", "/api/apps/call", 403, None),
    ("POST", "/api/apps/prepare", 403, None),
    ("POST", "/api/mcp/signin", 403, None),
    ("POST", "/api/mcp/signout", 403, None),
    ("POST", "/api/settings", 403, None),  # includes the MCP servers text
    ("POST", "/api/forge/jobs", 403, None),
    ("DELETE", "/api/forge/jobs/abc", 403, None),
    ("POST", "/api/tuning/start", 403, None),
    ("POST", "/api/quality/kld", 403, None),
    ("POST", "/api/update/apply", 403, 428),
    ("POST", "/api/engines/install", 403, 428),
    ("POST", "/api/chats", None, None),  # chatting is allowed
    ("POST", "/api/attachments", None, None),
    ("POST", "/api/bench/start", None, None),
    ("POST", "/api/flags/explain", None, None),
    ("GET", "/api/tools", None, None),
    ("GET", "/app-frame/abcdefghijklmnopqrstu", None, None),
    ("POST", "/api/hosts", 403, 403),  # never, not even for an admin
]


def test_every_write_route_of_the_other_parts_needs_admin_rights() -> None:
    from fle_studio import remote  # noqa: PLC0415

    for method, path, default, admin in WRITE_ROUTES_OF_OTHER_PARTS:
        got = remote.check_rights({"admin": False}, method, path, False)
        assert (got[0] if got else None) == default, (method, path, got)
        got = remote.check_rights({"admin": True}, method, path, False)
        assert (got[0] if got else None) == admin, (method, path, got)


def test_remote_admin_paths_are_not_passed_on_not_even_the_bare_one_or_dotdot(engine_host: Rig, rigs: Callable[..., Rig]) -> None:
    """`/api/remote` (without a slash after it) and `..` or percent-encoded dot segments must not slip past the guard that keeps
    pairing and host management from being driven through another host."""
    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    base = f"/h/{host['id']}"
    # the bare path is the one that really slipped through the old prefix test ("/api/remote/" needs the slash)
    assert client.owner.call("GET", base + "/api/remote")[1]["error"]["message"] == "This path is not passed on to a host."
    for path in ("/api/remote", "/api/remote?x=1", "/api/hosts", "/api/../api/remote/clients", "/api/%2e%2e/api/remote/code", "/api/./remote/code",
                 "/api/state/../remote/code", "/v1/../api/remote"):
        status, body = client.owner.call("GET", base + path)
        assert status in (403, 404), (path, status, body)  # 403 = this check, 404 = this Studio's own router
        if status == 403:
            assert body["error"]["message"] == "This path is not passed on to a host.", (path, body)


def test_large_answers_are_streamed_not_buffered(engine_host: Rig, rigs: Callable[..., Rig], monkeypatch) -> None:
    from fle_studio import remote

    client = rigs("client", False)
    host = add_host(client, engine_host)["host"]
    monkeypatch.setattr(remote, "_BUFFER_MAX", 10)  # every answer with a body counts as large
    saved: list[str] = []
    monkeypatch.setattr(remote, "_mirror_save", lambda *a, **k: saved.append("x"))
    status, body = client.owner.call("GET", f"/h/{host['id']}/api/state")
    assert status == 200 and body["data_dir"] == str(engine_host.studio.data_dir)  # still complete and valid
    assert saved == []  # not read whole, so not mirrored either
