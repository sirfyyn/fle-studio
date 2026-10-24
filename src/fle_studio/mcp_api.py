"""HTTP side of MCP sign-in and MCP Apps: handlers added to the server's Handler (see ``install``), the same way as
chat_api.py.

Sign-in: ``POST /api/mcp/signin`` returns the address to open in the browser; the server sends the browser back to
``GET /oauth/callback`` on Studio's own loopback port; ``POST /api/mcp/signout`` removes the token.

Apps (MCP Apps, SEP-1865, stable 2026-01-26): ``POST /api/apps/prepare`` reads a tool's ``ui://`` resource and keeps it
under a random frame id; ``GET /app-frame/<id>`` serves it with its own Content-Security-Policy, built from the
resource's metadata, into an ``<iframe sandbox="allow-scripts">`` (no allow-same-origin, so the document has an opaque
origin and cannot reach Studio). The app talks to the page through postMessage only; ``POST /api/apps/call`` is where
the page forwards an app's ``tools/call`` to the app's own server.
"""

from __future__ import annotations

import html
import re
import secrets
import threading
import time
from typing import Any

from . import chat_api, tools
from .mcp import McpError

ROUTES: list[tuple[str, str, str]] = [
    ("POST", r"^/api/mcp/signin$", "mcp_signin"),
    ("POST", r"^/api/mcp/signout$", "mcp_signout"),
    ("GET", r"^/oauth/callback$", "oauth_callback"),
    ("POST", r"^/api/apps/prepare$", "apps_prepare"),
    ("POST", r"^/api/apps/call$", "apps_call"),
    ("GET", r"^/app-frame/(?P<frame>[A-Za-z0-9_-]{20,64})$", "app_frame"),
]

_ApiError: Any = None
FRAME_LIMIT = 64
FRAME_SECONDS = 6 * 3600
_frames: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

ORIGIN = re.compile(
    r"^(?:(?:https|wss)://(?:\*\.)?[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::\d{1,5})?"
    r"|(?:http|ws)://(?:127\.0\.0\.1|localhost)(?::\d{1,5})?)$"
)


def _fail(status: int, message: str) -> Exception:
    return _ApiError(status, message)  # type: ignore[no-any-return]


def origins(value: Any) -> list[str]:
    """The entries of a CSP domain list that are plain origins; anything else (quotes, semicolons, ``*``) is dropped."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and ORIGIN.match(v)][:30]


def frame_csp(meta: dict[str, Any]) -> str:
    """The Content-Security-Policy of an app's document, from ``_meta.ui.csp`` of its resource.

    Without metadata the app may run its own inline scripts and styles, show images and media from ``data:`` URLs and
    nothing else: no network, no frames, no forms. Domains the resource asks for are added to the matching
    directives; ``'self'`` is left out everywhere because the frame's own origin is opaque and Studio's must stay out.
    """
    ui = meta.get("ui") if isinstance(meta.get("ui"), dict) else {}
    csp = ui.get("csp") if isinstance(ui.get("csp"), dict) else {}
    resources = " ".join(origins(csp.get("resourceDomains")))
    connect = " ".join(origins(csp.get("connectDomains"))) or "'none'"
    frames = " ".join(origins(csp.get("frameDomains"))) or "'none'"
    base = " ".join(origins(csp.get("baseUriDomains"))) or "'none'"
    parts = [
        "default-src 'none'",
        f"script-src 'unsafe-inline' {resources}".strip(),
        f"style-src 'unsafe-inline' {resources}".strip(),
        f"img-src data: {resources}".strip(),
        f"font-src data: {resources}".strip(),
        f"media-src data: {resources}".strip(),
        f"connect-src {connect}",
        f"frame-src {frames}",
        f"base-uri {base}",
        "object-src 'none'",
        "form-action 'none'",
        "frame-ancestors 'self'",
    ]
    return "; ".join(parts)


def _new_frame(record: dict[str, Any]) -> str:
    frame = secrets.token_urlsafe(24)
    now = time.time()
    with _lock:
        for k in [k for k, v in _frames.items() if now - v["at"] > FRAME_SECONDS]:
            del _frames[k]
        while len(_frames) >= FRAME_LIMIT:
            del _frames[min(_frames, key=lambda k: _frames[k]["at"])]
        _frames[frame] = {**record, "at": now}
    return frame


def _frame(frame: str) -> dict[str, Any]:
    with _lock:
        rec = _frames.get(frame)
    if rec is None:
        raise _fail(404, "This app is no longer available. Run the tool again.")
    return rec


# ------------------------------------------------------------------------------------------------ sign-in


def redirect_uri(handler: Any) -> str:
    return f"http://127.0.0.1:{handler.server.server_address[1]}/oauth/callback"


def api_mcp_signin(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    box = chat_api.toolbox(self.studio)
    started = box.begin_sign_in(str(body.get("server") or ""), redirect_uri(self))
    self.send_json(200, {"auth_url": started["auth_url"], "issuer": started["issuer"], "scope": started["scope"]})


def api_mcp_signout(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    box = chat_api.toolbox(self.studio)
    removed = box.sign_out(str(body.get("server") or ""))
    self.send_json(200, {"signed_out": True, "had_token": removed})


_CALLBACK_PAGE = """<!doctype html><meta charset="utf-8"><title>FLE Studio</title>
<style>body{font:16px system-ui,sans-serif;max-width:32rem;margin:15vh auto;padding:0 1rem;color:#222}
@media (prefers-color-scheme:dark){body{background:#16181c;color:#e6e6e6}}h1{font-size:1.2rem}</style>
<h1>__TITLE__</h1><p>__TEXT__</p>"""


def api_oauth_callback(self: Any, query: dict[str, str]) -> None:
    box = chat_api.toolbox(self.studio)
    try:
        name = box.finish_sign_in(
            query.get("state", ""), query.get("code", ""), query.get("error", ""), query.get("error_description", "")
        )
        title, text, status = (
            "Signed in",
            f"{name} is connected. You can close this tab and go back to FLE Studio.",
            200,
        )
    except RuntimeError as exc:  # OAuthError
        title, text, status = "Sign-in did not work", str(exc), 400
    page = _CALLBACK_PAGE.replace("__TITLE__", html.escape(title)).replace("__TEXT__", html.escape(text))
    self._send(
        status,
        page.encode(),
        "text/html; charset=utf-8",
        {
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'",
            "X-Frame-Options": "DENY",
        },
    )


# ------------------------------------------------------------------------------------------------ apps


def api_apps_prepare(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    box = chat_api.toolbox(self.studio)
    name = str(body.get("name") or "")
    tool = next((t for t in box.tools() if t.name == name and t.source == "mcp"), None)
    if tool is None or not tool.ui:
        raise _fail(404, "That tool has no app.")
    server = box.mcp.get(tool.server)
    if server is None:
        raise _fail(404, "That tool's server is not connected.")
    try:
        resource = server.read_resource(tool.ui)
    except McpError as exc:
        raise _fail(502, str(exc)) from None
    meta = resource["meta"]
    ui = meta.get("ui") if isinstance(meta.get("ui"), dict) else {}
    frame = _new_frame(
        {
            "html": resource["html"],
            "csp": frame_csp(meta),
            "server": tool.server,
            "uri": tool.ui,
            "tool": tool.remote_name,
            "wire_name": tool.name,
        }
    )
    self.send_json(
        200,
        {
            "frame": frame,
            "url": f"/app-frame/{frame}",
            "server": tool.server,
            "tool": tool.remote_name,
            "uri": tool.ui,
            "prefers_border": ui.get("prefersBorder") if isinstance(ui.get("prefersBorder"), bool) else None,
            "trusted": tool.server in box.trusted(),
        },
    )


def api_app_frame(self: Any, query: dict[str, str], frame: str) -> None:
    rec = _frame(frame)
    self._send(
        200,
        rec["html"].encode("utf-8"),
        "text/html; charset=utf-8",
        {
            "Content-Security-Policy": rec["csp"],
            "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=(), clipboard-write=()",
            "Cross-Origin-Resource-Policy": "same-origin",
        },
    )


def api_apps_call(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    rec = _frame(str(body.get("frame") or ""))
    box = chat_api.toolbox(self.studio)
    try:
        result = box.app_call(
            rec["server"],
            str(body.get("name") or ""),
            body.get("arguments") or {},
            approved=body.get("approved") is True,
        )
    except tools.NeedsApproval:
        raise _fail(409, "This tool needs your approval first.") from None
    except KeyError:
        raise _fail(404, "That tool is not available to this app (the server it came from may have gone).") from None
    self.send_json(200, {"result": result})


def install(handler: type, routes: list[Any], api_error: type) -> None:
    global _ApiError
    _ApiError = api_error
    for method, pattern, name in ROUTES:
        routes.append((method, re.compile(pattern), name))
        setattr(handler, f"api_{name}", globals()[f"api_{name}"])
