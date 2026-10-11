from __future__ import annotations

import http.client
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from chat_support import FAKE_MCP, clean_env, free_port

from fle_studio import mcp, oauth

REDIRECT = "http://127.0.0.1:9/oauth/callback"


class Fake:
    def __init__(self, proc: subprocess.Popen[bytes], port: int) -> None:
        self.proc, self.port = proc, port
        self.url = f"http://127.0.0.1:{port}/mcp"

    def state(self) -> dict[str, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/_state")
        doc = json.loads(conn.getresponse().read())
        conn.close()
        return doc  # type: ignore[no-any-return]

    def events(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.state()["log"] if e.get("event") == name]


def start(*extra: str, env: dict[str, str] | None = None) -> Fake:
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, str(FAKE_MCP), "--http", str(port), *extra],
        env=clean_env(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stderr is not None
    proc.stderr.readline()
    return Fake(proc, port)


def stop(fake: Fake) -> None:
    fake.proc.kill()
    fake.proc.wait(timeout=10)
    if fake.proc.stderr:
        fake.proc.stderr.close()


@pytest.fixture
def oauth_server() -> Iterator[Fake]:
    fake = start("--oauth")
    try:
        yield fake
    finally:
        stop(fake)


def approve(auth_url: str) -> tuple[str, str]:
    """What the browser does: open the authorization address, follow the redirect; (state, code)."""
    u = urlparse(auth_url)
    conn = http.client.HTTPConnection(u.hostname or "", u.port, timeout=10)
    conn.request("GET", u.path + "?" + u.query)
    resp = conn.getresponse()
    resp.read()
    assert resp.status == 302, resp.status
    back = parse_qs(urlparse(resp.getheader("Location") or "").query)
    conn.close()
    return back["state"][0], back["code"][0]


def sign_in(manager: oauth.OAuthManager, fake: Fake, name: str = "web", **kw: Any) -> None:
    started = manager.begin(name, fake.url, REDIRECT, **kw)
    state, code = approve(started["auth_url"])
    assert manager.complete(state, code) == name


# ------------------------------------------------------------------------------------------------ pure pieces


def test_parse_challenges_reads_schemes_and_parameters() -> None:
    header = (
        'Bearer realm="mcp", resource_metadata="https://x.example/.well-known/oauth-protected-resource/mcp", '
        'scope="a b", error="insufficient_scope", Basic realm="other \\"one\\""'
    )
    got = oauth.parse_challenges(header)
    assert [c["scheme"] for c in got] == ["Bearer", "Basic"]
    assert got[0]["params"]["scope"] == "a b"
    assert got[0]["params"]["resource_metadata"].endswith("/oauth-protected-resource/mcp")
    assert got[1]["params"]["realm"] == 'other "one"'
    assert oauth.bearer_challenge(header)["error"] == "insufficient_scope"
    assert oauth.bearer_challenge("Basic realm=x") == {} and oauth.bearer_challenge("") == {}


def test_canonical_resource() -> None:
    assert oauth.canonical_resource("HTTPS://Example.COM:443/mcp/") == "https://example.com/mcp"
    assert oauth.canonical_resource("https://example.com/") == "https://example.com"
    assert oauth.canonical_resource("http://127.0.0.1:8080/mcp#frag") == "http://127.0.0.1:8080/mcp"
    assert oauth.canonical_resource("https://example.com:8443/a/b") == "https://example.com:8443/a/b"


def test_pkce_matches_the_rfc_example() -> None:
    # RFC 7636, appendix B
    assert oauth.challenge_s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == (
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )
    a, b = oauth.make_verifier(), oauth.make_verifier()
    assert (
        a != b
        and 43 <= len(a) <= 128
        and set(a) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
    )


def test_metadata_addresses_follow_the_specification_order() -> None:
    assert oauth.authorization_server_urls("https://auth.example/tenant1") == [
        "https://auth.example/.well-known/oauth-authorization-server/tenant1",
        "https://auth.example/.well-known/openid-configuration/tenant1",
        "https://auth.example/tenant1/.well-known/openid-configuration",
    ]
    assert oauth.authorization_server_urls("https://auth.example") == [
        "https://auth.example/.well-known/oauth-authorization-server",
        "https://auth.example/.well-known/openid-configuration",
    ]
    assert oauth.protected_resource_urls("https://x.example/a/mcp", {}) == [
        "https://x.example/.well-known/oauth-protected-resource/a/mcp",
        "https://x.example/.well-known/oauth-protected-resource",
    ]
    assert oauth.protected_resource_urls("https://x.example/mcp", {"resource_metadata": "https://m.example/p"}) == [
        "https://m.example/p"
    ]


def test_plain_http_is_refused_except_for_loopback() -> None:
    oauth.require_secure("https://x.example/a", "thing")
    oauth.require_secure("http://127.0.0.1:5/a", "thing")
    oauth.require_secure("http://localhost/a", "thing")
    with pytest.raises(oauth.OAuthError):
        oauth.require_secure("http://x.example/a", "thing")


def test_authorization_url_carries_pkce_state_and_resource() -> None:
    meta = {"authorization_endpoint": "https://a.example/authorize"}
    url = oauth.authorization_url(
        meta,
        client_id="c1",
        redirect_uri=REDIRECT,
        state="st",
        verifier="v" * 43,
        resource="https://x.example/mcp",
        scope="read write",
    )
    q = parse_qs(urlparse(url).query)
    assert q["response_type"] == ["code"] and q["code_challenge_method"] == ["S256"]
    assert q["code_challenge"] == [oauth.challenge_s256("v" * 43)]
    assert q["resource"] == ["https://x.example/mcp"] and q["state"] == ["st"] and q["scope"] == ["read write"]
    assert "code_verifier" not in url and "v" * 43 not in url


# ------------------------------------------------------------------------------------------------ token storage


def test_token_files_are_private_and_hold_no_server_name(tmp_path: Path) -> None:
    store = oauth.TokenStore(tmp_path)
    store.save("my-server", "https://x.example/mcp/", {"access_token": "secret-1"})
    path = store.path("my-server", "https://x.example/mcp")
    assert path.parent == tmp_path / "oauth" and "my-server" not in path.name
    assert store.load("my-server", "https://x.example/mcp") == {"access_token": "secret-1"}
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    store.save("my-server", "https://x.example/mcp", {"access_token": "secret-2"})  # replaced, not appended
    assert store.load("my-server", "https://x.example/mcp") == {"access_token": "secret-2"}
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temporary file left behind
    # another address of the same name is another token
    assert store.load("my-server", "https://elsewhere.example/mcp") is None
    assert store.delete("my-server", "https://x.example/mcp") is True
    assert (
        store.load("my-server", "https://x.example/mcp") is None
        and store.delete("my-server", "https://x.example/mcp") is False
    )


# ------------------------------------------------------------------------------------------------ the whole flow


def test_discovery_reads_both_metadata_documents(oauth_server: Fake) -> None:
    disc = oauth.discover(
        oauth_server.url,
        {
            "resource_metadata": f"http://127.0.0.1:{oauth_server.port}/.well-known/oauth-protected-resource/mcp",
            "scope": "mcp:tools",
        },
    )
    assert disc.resource == oauth_server.url and disc.issuer == f"http://127.0.0.1:{oauth_server.port}"
    assert disc.scope == "mcp:tools" and "S256" in disc.metadata["code_challenge_methods_supported"]
    # without the header, the well-known address with the path is found, and the scope comes from scopes_supported
    again = oauth.discover(oauth_server.url, {})
    assert again.scope == "mcp:tools" and again.steps[0].endswith("oauth-protected-resource/mcp -> 200")


def test_sign_in_with_registration_stores_a_token_and_the_server_accepts_it(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    spec = {"transport": "http", "url": oauth_server.url, "headers": {}}
    anon = mcp.McpServer("web", spec, 20, manager)
    anon.connect()
    assert anon.status == "auth_required" and anon.challenge["resource_metadata"].endswith("/mcp")
    assert manager.status("web", oauth_server.url) == {"signed_in": False}

    started = manager.begin("web", oauth_server.url, REDIRECT, challenge=anon.challenge)
    assert started["scope"] == "mcp:tools"
    state, code = approve(started["auth_url"])
    assert manager.complete(state, code) == "web"
    assert manager.status("web", oauth_server.url)["signed_in"] is True

    signed = mcp.McpServer("web", spec, 20, manager)
    signed.connect()
    try:
        assert signed.status == "ready", signed.error
        assert signed.call("echo", {"text": "hi"}) == ("echo: hi", False)
    finally:
        signed.close()

    registered = oauth_server.events("register")
    assert registered[0]["method"] == "none" and registered[0]["redirects"] == [REDIRECT]
    authorized = oauth_server.events("authorize")
    assert (
        authorized[0]["ok"] and authorized[0]["resource"] == oauth_server.url
    )  # RFC 8707 in the authorization request
    token = oauth_server.events("token")[0]
    assert token["ok"] and token["resource"] == oauth_server.url  # and in the token request
    assert not any(e.get("token_in_query") for e in oauth_server.events("mcp-request"))


def test_the_token_is_not_in_the_status_and_not_in_other_places(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    sign_in(manager, oauth_server)
    status = manager.status("web", oauth_server.url)
    assert set(status) == {"signed_in", "scope", "issuer", "expires_at", "can_refresh"}
    secret = manager.access_token("web", oauth_server.url)
    assert secret not in json.dumps(status)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].parent.name == "oauth"


def test_a_token_that_is_about_to_expire_is_refreshed_and_rotated(tmp_path: Path) -> None:
    fake = start("--oauth", env={"FAKE_MCP_EXPIRES": "30"})  # shorter than the refresh margin
    try:
        manager = oauth.OAuthManager(tmp_path)
        sign_in(manager, fake)
        first = manager.store.load("web", fake.url)
        assert first is not None
        token = manager.access_token("web", fake.url)
        again = manager.store.load("web", fake.url)
        assert again is not None
        assert token == again["access_token"] and token != first["access_token"]
        assert again["refresh_token"] != first["refresh_token"]  # rotated
        grants = [e["grant"] for e in fake.events("token")]
        assert grants == ["authorization_code", "refresh_token"]
        assert fake.events("token")[1]["resource"] == fake.url
    finally:
        stop(fake)


def test_a_401_with_a_refresh_token_repeats_the_request_once(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    sign_in(manager, oauth_server)
    spec = {"transport": "http", "url": oauth_server.url, "headers": {}}
    srv = mcp.McpServer("web", spec, 20, manager)
    srv.connect()
    try:
        assert srv.status == "ready", srv.error
        rec = manager.store.load("web", oauth_server.url)
        assert rec is not None
        rec["access_token"] = "revoked-token"  # the server does not know this one any more
        manager.store.save("web", oauth_server.url, rec)
        assert srv.call("echo", {"text": "after refresh"}) == ("echo: after refresh", False)
        assert [e["grant"] for e in oauth_server.events("token")] == ["authorization_code", "refresh_token"]
    finally:
        srv.close()


def test_a_refused_refresh_ends_the_sign_in(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    sign_in(manager, oauth_server)
    rec = manager.store.load("web", oauth_server.url)
    assert rec is not None
    rec.update(refresh_token="unknown", expires_at=time.time() - 5)
    manager.store.save("web", oauth_server.url, rec)
    with pytest.raises(oauth.NeedsSignIn):
        manager.access_token("web", oauth_server.url)
    assert manager.status("web", oauth_server.url) == {"signed_in": False}


def test_sign_out_deletes_the_token_and_asks_the_server_to_revoke(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    sign_in(manager, oauth_server)
    assert manager.sign_out("web", oauth_server.url) is True
    assert not list((tmp_path / "oauth").glob("*"))
    assert [e["hint"] for e in oauth_server.events("revoke")] == ["refresh_token", "access_token"]
    with pytest.raises(oauth.NeedsSignIn):
        manager.access_token("web", oauth_server.url)


def test_the_callback_checks_state_and_expiry(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    started = manager.begin("web", oauth_server.url, REDIRECT)
    state, code = approve(started["auth_url"])
    with pytest.raises(oauth.OAuthError, match="not known"):
        manager.complete("wrong-state", code)
    with pytest.raises(oauth.OAuthError, match="refused the sign-in"):
        manager.complete(state, "", error="access_denied")
    with pytest.raises(oauth.OAuthError, match="not known"):  # a state works once
        manager.complete(state, code)
    started = manager.begin("web", oauth_server.url, REDIRECT)
    state, code = approve(started["auth_url"])
    manager._pending[state]["at"] -= oauth.PENDING_SECONDS + 1
    with pytest.raises(oauth.OAuthError, match="not known"):
        manager.complete(state, code)


def test_a_wrong_code_is_refused_by_the_token_endpoint(oauth_server: Fake, tmp_path: Path) -> None:
    manager = oauth.OAuthManager(tmp_path)
    started = manager.begin("web", oauth_server.url, REDIRECT)
    with pytest.raises(oauth.OAuthError, match="token request was refused"):
        manager.complete(started["state"], "made-up-code")
    assert manager.status("web", oauth_server.url) == {"signed_in": False}


def test_without_registration_a_client_id_is_needed(tmp_path: Path) -> None:
    fake = start("--oauth", "--no-dcr")
    try:
        manager = oauth.OAuthManager(tmp_path)
        with pytest.raises(oauth.OAuthError, match='"client_id"'):
            manager.begin("web", fake.url, REDIRECT)
    finally:
        stop(fake)


def test_a_configured_client_id_is_used_instead_of_registering(tmp_path: Path) -> None:
    fake = start("--oauth", "--no-dcr")
    try:
        manager = oauth.OAuthManager(tmp_path)
        sign_in(manager, fake, preset={"client_id": "my-own-client", "scope": "mcp:tools"})
        assert manager.store.load("web", fake.url)["client_id"] == "my-own-client"  # type: ignore[index]
        assert fake.events("register") == []
    finally:
        stop(fake)


def test_discovery_refuses_a_server_without_s256() -> None:
    def http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], Any]:
        if "oauth-protected-resource" in url:
            return 200, {}, {"resource": "https://x.example/mcp", "authorization_servers": ["https://auth.example"]}
        return (
            200,
            {},
            {
                "issuer": "https://auth.example",
                "authorization_endpoint": "https://auth.example/a",
                "token_endpoint": "https://auth.example/t",
                "code_challenge_methods_supported": ["plain"],
            },
        )

    with pytest.raises(oauth.OAuthError, match="S256"):
        oauth.discover("https://x.example/mcp", {}, http)


def test_discovery_refuses_metadata_that_belongs_to_another_server() -> None:
    def http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], Any]:
        return 200, {}, {"resource": "https://evil.example/mcp", "authorization_servers": ["https://auth.example"]}

    with pytest.raises(oauth.OAuthError, match="does not belong"):
        oauth.discover("https://x.example/mcp", {}, http)


def test_discovery_refuses_a_plain_http_authorization_server() -> None:
    def http(method: str, url: str, **kw: Any) -> tuple[int, dict[str, str], Any]:
        return 200, {}, {"resource": "https://x.example/mcp", "authorization_servers": ["http://auth.example"]}

    with pytest.raises(oauth.OAuthError, match="https"):
        oauth.discover("https://x.example/mcp", {}, http)


# ------------------------------------------------------------------------------------------------ old SSE transport


def test_legacy_sse_server_is_used_when_the_post_is_refused() -> None:
    fake = start("--legacy")
    spec = {"transport": "http", "url": f"http://127.0.0.1:{fake.port}/sse", "headers": {}}
    # the URL of a legacy server is its /sse address: a POST there fails with 4xx, then the GET stream is opened
    srv = mcp.McpServer("old", spec, 20)
    try:
        srv.connect()
        assert srv.status == "ready", srv.error
        assert srv.fallback is True and isinstance(srv.transport, mcp.SseTransport)
        assert srv.call("echo", {"text": "old way"}) == ("echo: old way", False)
        assert srv.call("fail", {}) == ("boom", True)
    finally:
        srv.close()
        stop(fake)


def test_type_sse_goes_straight_to_the_event_stream() -> None:
    fake = start("--legacy")
    spec = {"transport": "sse", "url": f"http://127.0.0.1:{fake.port}/sse", "headers": {}}
    srv = mcp.McpServer("old", spec, 20)
    try:
        srv.connect()
        assert srv.status == "ready", srv.error
        assert srv.call("add", {"a": 1, "b": 2}) == ("3", False)
    finally:
        srv.close()
        stop(fake)


def test_a_non_4xx_failure_does_not_fall_back() -> None:
    srv = mcp.McpServer("gone", {"transport": "http", "url": f"http://127.0.0.1:{free_port()}/mcp", "headers": {}}, 5)
    srv.connect()
    assert srv.status == "error" and srv.fallback is False
