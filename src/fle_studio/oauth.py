"""OAuth for remote MCP servers, following the MCP authorization specification (revision 2025-11-25, which keeps the
flow of 2025-06-18 and adds Client ID Metadata Documents and step-up). Standard library only.

The flow, in the order it happens:

1. The server answers a request with ``401`` and ``WWW-Authenticate: Bearer resource_metadata="…", scope="…"``
   (:func:`parse_challenges`).
2. Protected Resource Metadata (RFC 9728) tells which authorization server to use: first the URL from the header, then
   ``/.well-known/oauth-protected-resource/<path>``, then ``/.well-known/oauth-protected-resource`` (:func:`discover`).
3. Authorization Server Metadata: RFC 8414 with path insertion, then OpenID Connect Discovery with path insertion and
   path appending, in the order the specification prescribes. A server without ``code_challenge_methods_supported``
   containing ``S256`` is refused.
4. A client id: one the user configured, else Dynamic Client Registration (RFC 7591) when the server offers it, else
   the user is asked for one. (Client ID Metadata Documents need a public HTTPS page that names Studio's loopback
   redirect; Studio has none of its own, but an ``https://`` client id that the user configured works like any other.)
5. Authorization code with PKCE (S256) in the user's browser, redirected to Studio's own loopback port. The
   ``resource`` parameter (RFC 8707) goes into the authorization and the token request; ``state`` is checked.
6. Tokens are kept in the data folder (``oauth/``, files with mode 0600), refreshed shortly before they expire, and
   removed by signing out. They never go into settings, logs or exports.

Every URL of the authorization server must be ``https`` (loopback ``http`` is accepted, for local servers and tests).
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from . import net

PENDING_SECONDS = 600  # how long a sign-in may take
EXPIRY_MARGIN = 60  # refresh a token that has less than this many seconds left
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
CLIENT_NAME = "FLE Studio"


class OAuthError(RuntimeError):
    """Signing in failed; the message says why in words the user can act on."""


class NeedsSignIn(OAuthError):
    """There is no usable token (none stored, or it was refused and cannot be refreshed)."""


# ------------------------------------------------------------------------------------------------ small helpers


def canonical_resource(url: str) -> str:
    """The canonical URI of an MCP server (RFC 8707): lower-case scheme and host, no fragment, no trailing slash."""
    u = urlparse(url.strip())
    host = (u.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host + (f":{u.port}" if u.port and u.port != {"http": 80, "https": 443}.get(u.scheme) else "")
    path = u.path.rstrip("/") if u.path != "/" else ""
    return urlunparse((u.scheme.lower(), netloc, path, "", u.query, ""))


def require_secure(url: str, what: str) -> None:
    u = urlparse(url)
    if u.scheme == "https" and u.hostname:
        return
    if u.scheme == "http" and (u.hostname or "").lower() in LOOPBACK:
        return
    raise OAuthError(f"The {what} ({url}) is not an https address; FLE Studio only signs in over https.")


def parse_challenges(header: str) -> list[dict[str, Any]]:
    """The challenges of a ``WWW-Authenticate`` header: [{"scheme": "Bearer", "params": {"resource_metadata": …}}]."""
    pattern = re.compile(
        r'\s*(?:(?P<param>[A-Za-z][\w-]*)\s*=\s*(?:"(?P<q>(?:[^"\\]|\\.)*)"|(?P<t>[^\s,]+))|(?P<scheme>[A-Za-z][\w-]*))'
        r"\s*,?"
    )
    out: list[dict[str, Any]] = []
    pos = 0
    while pos < len(header or ""):
        m = pattern.match(header, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        if m.group("scheme"):
            out.append({"scheme": m.group("scheme"), "params": {}})
        elif out:
            value = m.group("q") if m.group("q") is not None else m.group("t")
            out[-1]["params"][m.group("param").lower()] = re.sub(r"\\(.)", r"\1", value or "")
    return out


def bearer_challenge(header: str) -> dict[str, str]:
    """The parameters of the first Bearer challenge in a header (empty when there is none)."""
    for c in parse_challenges(header):
        if c["scheme"].lower() == "bearer":
            return dict(c["params"])
    return {}


def make_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(48)).rstrip(b"=").decode()  # 64 characters


def challenge_s256(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def http_request(
    method: str,
    url: str,
    *,
    form: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> tuple[int, dict[str, str], Any]:
    """(status, headers, parsed JSON or None). Never raises for an HTTP error status; raises OAuthError when the
    server cannot be reached. No redirect is followed to a different host."""
    sent = {"User-Agent": net.USER_AGENT, "Accept": "application/json", **(headers or {})}
    data = None
    if form is not None:
        data = urlencode(form).encode()
        sent["Content-Type"] = "application/x-www-form-urlencoded"
    elif body is not None:
        data = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=sent)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=net.ssl_context()) as r:
            status, head, raw = r.status, {k.lower(): v for k, v in r.headers.items()}, r.read(1_000_000)
    except urllib.error.HTTPError as exc:
        status, head, raw = exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read(100_000)
    except (OSError, ValueError) as exc:
        raise OAuthError(f"Could not reach {urlparse(url).hostname}: {getattr(exc, 'reason', exc)}") from None
    try:
        doc = json.loads(raw.decode("utf-8")) if raw else None
    except ValueError:
        doc = None
    return status, head, doc


Http = Callable[..., tuple[int, dict[str, str], Any]]


# ------------------------------------------------------------------------------------------------ discovery


@dataclass
class Discovery:
    resource: str  # canonical URI of the MCP server: the ``resource`` parameter
    resource_metadata: dict[str, Any]
    issuer: str
    metadata: dict[str, Any]  # authorization server metadata
    scope: str = ""  # what to ask for; empty = send no scope parameter
    steps: list[str] = field(default_factory=list)  # what was tried, for the error text and the tests


def protected_resource_urls(mcp_url: str, challenge: dict[str, str]) -> list[str]:
    if challenge.get("resource_metadata"):
        return [challenge["resource_metadata"]]
    u = urlparse(mcp_url)
    root = f"{u.scheme}://{u.netloc}/.well-known/oauth-protected-resource"
    path = u.path.rstrip("/")
    return ([root + path] if path else []) + [root]


def authorization_server_urls(issuer: str) -> list[str]:
    """The metadata URLs to try for an issuer, in the order of the MCP specification."""
    u = urlparse(issuer)
    base = f"{u.scheme}://{u.netloc}"
    path = u.path.rstrip("/")
    if path:
        return [
            f"{base}/.well-known/oauth-authorization-server{path}",
            f"{base}/.well-known/openid-configuration{path}",
            f"{base}{path}/.well-known/openid-configuration",
        ]
    return [f"{base}/.well-known/oauth-authorization-server", f"{base}/.well-known/openid-configuration"]


def discover(mcp_url: str, challenge: dict[str, str], http: Http = http_request) -> Discovery:
    """Find the authorization server of an MCP server and read its metadata (steps 2 and 3 of the module text)."""
    resource = canonical_resource(mcp_url)
    steps: list[str] = []
    rm: dict[str, Any] | None = None
    for url in protected_resource_urls(mcp_url, challenge):
        require_secure(url, "resource metadata address")
        status, _, doc = http("GET", url)
        steps.append(f"{url} -> {status}")
        if status == 200 and isinstance(doc, dict):
            rm = doc
            break
    if rm is None:
        # Servers of the 2025-03-26 revision publish no resource metadata and are their own authorization server.
        u = urlparse(mcp_url)
        rm = {"authorization_servers": [f"{u.scheme}://{u.netloc}"]}
        steps.append("no protected resource metadata; trying the server's own address as authorization server")
    elif rm.get("resource") and canonical_resource(str(rm["resource"])) != resource:
        raise OAuthError(
            f"The server describes itself as {rm['resource']}, not {resource}; refusing to sign in "
            "(the metadata does not belong to this server)."
        )
    servers = rm.get("authorization_servers")
    if not isinstance(servers, list) or not servers or not isinstance(servers[0], str):
        raise OAuthError("The server's resource metadata names no authorization server.")
    issuer = servers[0].rstrip("/") if urlparse(servers[0]).path in ("", "/") else servers[0]
    require_secure(issuer, "authorization server")
    meta: dict[str, Any] | None = None
    for url in authorization_server_urls(issuer):
        status, _, doc = http("GET", url)
        steps.append(f"{url} -> {status}")
        if status == 200 and isinstance(doc, dict) and doc.get("authorization_endpoint"):
            meta = doc
            break
    if meta is None:
        raise OAuthError(f"The authorization server {issuer} has no metadata at the well-known addresses.")
    if str(meta.get("issuer") or issuer).rstrip("/") != issuer.rstrip("/"):
        raise OAuthError(
            f"The metadata of {issuer} names a different issuer ({meta.get('issuer')}); refusing to go on."
        )
    for key in ("authorization_endpoint", "token_endpoint"):
        if not isinstance(meta.get(key), str):
            raise OAuthError(f"The authorization server metadata has no {key}.")
        require_secure(meta[key], key.replace("_", " "))
    if "S256" not in (meta.get("code_challenge_methods_supported") or []):
        raise OAuthError(
            "The authorization server does not say that it supports PKCE with S256 "
            "(code_challenge_methods_supported), so FLE Studio will not sign in."
        )
    supported = rm.get("scopes_supported")
    scope = challenge.get("scope") or (" ".join(str(s) for s in supported) if isinstance(supported, list) else "")
    return Discovery(resource, rm, issuer, meta, scope, steps)


def register_client(meta: dict[str, Any], redirect_uri: str, scope: str, http: Http = http_request) -> dict[str, str]:
    """Dynamic Client Registration (RFC 7591) as a public client. Returns {"client_id", "client_secret"?}."""
    endpoint = meta.get("registration_endpoint")
    if not isinstance(endpoint, str):
        raise OAuthError(
            "This server does not offer automatic client registration. Register FLE Studio with it yourself and "
            'add "oauth": {"client_id": "…"} to its entry in the MCP servers text. Studio\'s redirect address is '
            f"{redirect_uri}"
        )
    require_secure(endpoint, "registration endpoint")
    body: dict[str, Any] = {
        "client_name": CLIENT_NAME,
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    if scope:
        body["scope"] = scope
    status, _, doc = http("POST", endpoint, body=body)
    if status not in (200, 201) or not isinstance(doc, dict) or not doc.get("client_id"):
        why = (doc or {}).get("error_description") or (doc or {}).get("error") if isinstance(doc, dict) else ""
        raise OAuthError(f"Client registration was refused ({status}{': ' + str(why) if why else ''}).")
    out = {"client_id": str(doc["client_id"])}
    if doc.get("client_secret"):
        out["client_secret"] = str(doc["client_secret"])
    return out


def authorization_url(
    meta: dict[str, Any], *, client_id: str, redirect_uri: str, state: str, verifier: str, resource: str, scope: str
) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
        "code_challenge": challenge_s256(verifier),
        "code_challenge_method": "S256",
        "resource": resource,
    }
    if scope:
        params["scope"] = scope
    endpoint = meta["authorization_endpoint"]
    sep = "&" if urlparse(endpoint).query else "?"
    return endpoint + sep + urlencode(params)


# ------------------------------------------------------------------------------------------------ tokens on disk


class TokenStore:
    """One JSON file per server in ``<data>/oauth/``, readable by the user only (0600; on Windows the data folder is
    inside the user's profile). The file name is a hash of the server's name and address, so changing the address
    of a server never sends its old token to the new one."""

    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "oauth"
        self._lock = threading.Lock()

    def path(self, name: str, resource: str) -> Path:
        digest = hashlib.sha256(f"{name}\n{canonical_resource(resource)}".encode()).hexdigest()[:20]
        return self.dir / f"{digest}.json"

    def load(self, name: str, resource: str) -> dict[str, Any] | None:
        try:
            doc = json.loads(self.path(name, resource).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def save(self, name: str, resource: str, record: dict[str, Any]) -> None:
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                os.chmod(self.dir, 0o700)
            target = self.path(name, resource)
            tmp = target.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(record, fh)
            except BaseException:
                with contextlib.suppress(OSError):
                    tmp.unlink()
                raise
            if os.name != "nt":
                os.chmod(tmp, 0o600)
            os.replace(tmp, target)

    def delete(self, name: str, resource: str) -> bool:
        with self._lock:
            try:
                self.path(name, resource).unlink()
                return True
            except OSError:
                return False


# ------------------------------------------------------------------------------------------------ the flows


def _token_record(doc: dict[str, Any], old: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(doc.get("access_token"), str) or not doc["access_token"]:
        raise OAuthError("The token response has no access token.")
    if str(doc.get("token_type") or "Bearer").lower() != "bearer":
        raise OAuthError(f"The server issued a {doc.get('token_type')} token; only Bearer tokens are supported.")
    expires_in = doc.get("expires_in")
    return {
        "access_token": doc["access_token"],
        "refresh_token": doc.get("refresh_token") or (old or {}).get("refresh_token") or "",
        "expires_at": time.time() + float(expires_in) if isinstance(expires_in, (int, float)) else 0.0,
        "scope": str(doc.get("scope") or (old or {}).get("scope") or ""),
    }


class OAuthManager:
    """Sign-ins in progress, tokens, refreshing. One per Studio. ``redirect_uri`` is Studio's own callback address."""

    def __init__(self, data_dir: Path, http: Http = http_request) -> None:
        self.store = TokenStore(data_dir)
        self.http = http
        self._pending: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    # ---- sign in

    def begin(
        self,
        name: str,
        url: str,
        redirect_uri: str,
        *,
        challenge: dict[str, str] | None = None,
        preset: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Start a sign-in. ``preset`` is {"client_id", "client_secret"?, "scope"?} from the server's settings.
        Returns {"auth_url", "state", "issuer", "scope"} to open in the browser."""
        preset = preset or {}
        disc = discover(url, challenge or {}, self.http)
        scope = preset.get("scope") or disc.scope
        if preset.get("client_id"):
            client = {
                "client_id": preset["client_id"],
                **({"client_secret": preset["client_secret"]} if preset.get("client_secret") else {}),
            }
        else:
            client = register_client(disc.metadata, redirect_uri, scope, self.http)
        state = secrets.token_urlsafe(24)
        verifier = make_verifier()
        auth_url = authorization_url(
            disc.metadata,
            client_id=client["client_id"],
            redirect_uri=redirect_uri,
            state=state,
            verifier=verifier,
            resource=disc.resource,
            scope=scope,
        )
        with self._lock:
            now = time.time()
            self._pending = {k: v for k, v in self._pending.items() if now - v["at"] < PENDING_SECONDS}
            self._pending[state] = {
                "at": now,
                "name": name,
                "url": url,
                "verifier": verifier,
                "redirect_uri": redirect_uri,
                "client": client,
                "issuer": disc.issuer,
                "token_endpoint": disc.metadata["token_endpoint"],
                "revocation_endpoint": disc.metadata.get("revocation_endpoint") or "",
                "resource": disc.resource,
                "scope": scope,
            }
        return {"auth_url": auth_url, "state": state, "issuer": disc.issuer, "scope": scope}

    def complete(self, state: str, code: str = "", error: str = "", description: str = "") -> str:
        """The browser came back to Studio's callback. Exchanges the code and stores the token; returns the server's
        name. Raises OAuthError with a readable message for anything that is not a valid, matching answer."""
        with self._lock:
            pend = self._pending.pop(state, None) if state else None
        if pend is None or time.time() - pend["at"] > PENDING_SECONDS:
            raise OAuthError(
                "This sign-in is not known here (it may have expired or been opened twice). Start it again."
            )
        if error:
            raise OAuthError(f"The server refused the sign-in: {error}{' - ' + description if description else ''}")
        if not code:
            raise OAuthError("The server's answer has no authorization code.")
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": pend["redirect_uri"],
            "client_id": pend["client"]["client_id"],
            "code_verifier": pend["verifier"],
            "resource": pend["resource"],
        }
        if pend["client"].get("client_secret"):
            form["client_secret"] = pend["client"]["client_secret"]
        status, _, doc = self.http("POST", pend["token_endpoint"], form=form)
        if status != 200 or not isinstance(doc, dict):
            why = (doc or {}).get("error_description") or (doc or {}).get("error") if isinstance(doc, dict) else ""
            raise OAuthError(f"The token request was refused ({status}{': ' + str(why) if why else ''}).")
        record = _token_record(doc)
        record.update(
            client_id=pend["client"]["client_id"],
            client_secret=pend["client"].get("client_secret", ""),
            token_endpoint=pend["token_endpoint"],
            revocation_endpoint=pend["revocation_endpoint"],
            issuer=pend["issuer"],
            resource=pend["resource"],
            saved_at=time.time(),
        )
        self.store.save(pend["name"], pend["url"], record)
        return str(pend["name"])

    # ---- tokens

    def status(self, name: str, url: str) -> dict[str, Any]:
        """What the page may know: no token, only whether there is one."""
        rec = self.store.load(name, url)
        if not rec:
            return {"signed_in": False}
        exp = float(rec.get("expires_at") or 0)
        return {
            "signed_in": True,
            "scope": rec.get("scope") or "",
            "issuer": rec.get("issuer") or "",
            "expires_at": exp or None,
            "can_refresh": bool(rec.get("refresh_token")),
        }

    def access_token(self, name: str, url: str) -> str:
        """A token for a request. Refreshes one that is about to expire. Raises NeedsSignIn when there is none."""
        rec = self.store.load(name, url)
        if not rec or not rec.get("access_token"):
            raise NeedsSignIn("Sign in to use this server.")
        exp = float(rec.get("expires_at") or 0)
        if exp and exp - time.time() < EXPIRY_MARGIN:
            rec = self.refresh(name, url)
        return str(rec["access_token"])

    def refresh(self, name: str, url: str) -> dict[str, Any]:
        """Use the refresh token. Raises NeedsSignIn (and forgets the token) when the server refuses it."""
        with self._refresh_lock:
            rec = self.store.load(name, url)
            if not rec:
                raise NeedsSignIn("Sign in to use this server.")
            if not rec.get("refresh_token"):
                self.store.delete(name, url)
                raise NeedsSignIn("The sign-in expired. Sign in again.")
            form = {
                "grant_type": "refresh_token",
                "refresh_token": rec["refresh_token"],
                "client_id": rec["client_id"],
                "resource": rec["resource"],
            }
            if rec.get("client_secret"):
                form["client_secret"] = rec["client_secret"]
            status, _, doc = self.http("POST", rec["token_endpoint"], form=form)
            if status != 200 or not isinstance(doc, dict):
                if 400 <= status < 500:
                    self.store.delete(name, url)
                    raise NeedsSignIn("The sign-in expired. Sign in again.")
                raise OAuthError(f"The token server did not answer the refresh ({status}); try again in a moment.")
            new = {**rec, **_token_record(doc, rec), "saved_at": time.time()}
            self.store.save(name, url, new)
            return new

    def sign_out(self, name: str, url: str) -> bool:
        """Remove the token; asks the server to revoke it when it has a revocation endpoint (best effort)."""
        rec = self.store.load(name, url)
        if rec and rec.get("revocation_endpoint") and rec.get("access_token"):
            for token, hint in ((rec.get("refresh_token"), "refresh_token"), (rec.get("access_token"), "access_token")):
                if token:
                    with contextlib.suppress(OAuthError):
                        form = {"token": token, "token_type_hint": hint, "client_id": rec["client_id"]}
                        if rec.get("client_secret"):
                            form["client_secret"] = rec["client_secret"]
                        self.http("POST", rec["revocation_endpoint"], form=form, timeout=8.0)
        return self.store.delete(name, url)


class Provider:
    """What one MCP server connection uses: the current token, and a way to ask for a fresh one after a 401."""

    def __init__(self, manager: OAuthManager, name: str, url: str) -> None:
        self.manager, self.name, self.url = manager, name, url

    def header(self) -> str | None:
        try:
            return f"Bearer {self.manager.access_token(self.name, self.url)}"
        except NeedsSignIn:
            return None

    def refresh(self) -> bool:
        """After a 401: True when a new token is there and the request is worth repeating."""
        try:
            self.manager.refresh(self.name, self.url)
            return True
        except OAuthError:
            return False


def callback_params(query: str) -> dict[str, str]:
    return {k: v for k, v in parse_qsl(query, keep_blank_values=True)}


__all__ = [
    "NeedsSignIn",
    "OAuthError",
    "OAuthManager",
    "Provider",
    "TokenStore",
    "Discovery",
    "authorization_server_urls",
    "authorization_url",
    "bearer_challenge",
    "callback_params",
    "canonical_resource",
    "challenge_s256",
    "discover",
    "http_request",
    "make_verifier",
    "parse_challenges",
    "protected_resource_urls",
    "register_client",
    "require_secure",
    "urljoin",
]
