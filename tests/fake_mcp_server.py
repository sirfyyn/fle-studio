#!/usr/bin/env python3
"""A tiny MCP server for tests (standard library only; macOS, Linux and Windows).

    python fake_mcp_server.py                     speak MCP over stdin/stdout (one JSON message per line)
    python fake_mcp_server.py --http PORT         Streamable HTTP on 127.0.0.1:PORT/mcp (JSON answers)
    python fake_mcp_server.py --http PORT --sse   the same, but tools/call answers as an event stream
    python fake_mcp_server.py --http PORT --oauth the same, but /mcp wants a Bearer token from the OAuth stand-in that
                                                  runs on the same port (discovery, registration, authorization with
                                                  PKCE, token, refresh, revocation); add --no-dcr to offer no
                                                  registration endpoint
    python fake_mcp_server.py --http PORT --legacy the old HTTP+SSE transport: POST /mcp is refused (405), GET /sse
                                                  opens an event stream and names the POST address

Tools: echo, add (read-only hint), write_note (changes state), fail (isError result), crash (exits without answering),
show_chart (an MCP App: its result is shown by ui://fake/chart), app_bump (callable by apps only).
Environment: FAKE_MCP_PAGE=N lists N tools per page (to test paging); FAKE_MCP_TOKEN=T makes the HTTP mode demand
``Authorization: Bearer T``; FAKE_MCP_EXPIRES=S makes OAuth access tokens last S seconds (default 3600).
GET /_state shows what the OAuth stand-in saw, as JSON.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

PROTOCOL = "2025-06-18"
NOTES: list[str] = []
BUMPS = [0]
APP_URI = "ui://fake/chart"
APP_MIME = "text/html;profile=mcp-app"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Repeat the text back.",
        "inputSchema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "annotations": {"readOnlyHint": True},
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "write_note",
        "description": "Save a note (changes state).",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    },
    {"name": "fail", "description": "Always reports an error.", "inputSchema": {"type": "object", "properties": {}}},
    {"name": "crash", "description": "Exits without answering.", "inputSchema": {"type": "object", "properties": {}}},
    {
        "name": "show_chart",
        "description": "Show a small chart of three numbers (an app).",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string"}}},
        "_meta": {"ui": {"resourceUri": APP_URI}},
    },
    {
        "name": "app_bump",
        "description": "Count one more click (for the app only).",
        "inputSchema": {"type": "object", "properties": {}},
        "_meta": {"ui": {"resourceUri": APP_URI, "visibility": ["app"]}},
    },
]

APP_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Fake chart</title>
<style>
:root { color-scheme: light dark; }
body { font: 14px system-ui, sans-serif; margin: 0; padding: 12px; background: #fff; color: #111; }
html[data-theme="dark"] body { background: #1b1d22; color: #eee; }
button { font: inherit; margin-right: 6px; }
#bars { display: flex; gap: 6px; align-items: flex-end; height: 80px; margin: 8px 0; }
#bars i { display: block; width: 28px; background: #3b82f6; }
</style></head><body>
<h3 id="title">waiting</h3>
<div id="bars"></div>
<p>theme: <b id="theme">?</b> | input: <b id="input">?</b> | bumps: <b id="bump">0</b> | answer: <b id="reply">-</b></p>
<button id="b1">Bump</button><button id="b2">Tell the chat</button><button id="b3">Open link</button>
<button id="b4">Write note</button>
<script>
let nextId = 1; const waiting = {};
function send(m) { parent.postMessage(Object.assign({ jsonrpc: "2.0" }, m), "*"); }
function ask(method, params) {
  return new Promise((ok) => { const id = nextId++; waiting[id] = ok; send({ id, method, params }); });
}
function theme(ctx) { if (ctx && ctx.theme) { document.documentElement.dataset.theme = ctx.theme;
  document.getElementById("theme").textContent = ctx.theme; } }
window.addEventListener("message", (e) => {
  const m = e.data || {};
  if (m.id !== undefined && waiting[m.id] && !m.method) { waiting[m.id](m); delete waiting[m.id]; return; }
  if (m.method === "ui/notifications/tool-input") {
    document.getElementById("input").textContent = JSON.stringify((m.params || {}).arguments || {}); }
  if (m.method === "ui/notifications/tool-result") {
    const r = (m.params || {}).structuredContent || {};
    document.getElementById("title").textContent = r.title || "chart";
    const box = document.getElementById("bars"); box.textContent = "";
    for (const v of r.values || []) {
      const i = document.createElement("i"); i.style.height = v * 10 + "px"; box.append(i); }
  }
  if (m.method === "ui/notifications/host-context-changed") theme(m.params);
  if (m.method === "ui/resource-teardown") send({ id: m.id, result: {} });
});
document.getElementById("b1").onclick = async () => {
  const r = await ask("tools/call", { name: "app_bump", arguments: {} });
  const t = ((r.result || {}).structuredContent || {}).count;
  document.getElementById("bump").textContent = r.error ? "error: " + r.error.message : t;
};
document.getElementById("b2").onclick = () => ask("ui/message",
  { role: "user", content: [{ type: "text", text: "Hello from the app" }] })
  .then((r) => { document.getElementById("reply").textContent = r.error ? "refused" : "sent"; });
document.getElementById("b3").onclick = () => ask("ui/open-link", { url: "https://example.com/" })
  .then((r) => { document.getElementById("reply").textContent = r.error ? "refused" : "opened"; });
document.getElementById("b4").onclick = async () => {
  const r = await ask("tools/call", { name: "write_note", arguments: { text: "from the app" } });
  document.getElementById("reply").textContent = r.error ? "error: " + r.error.message
    : ((r.result.content || [{}])[0].text || "");
};
(async () => {
  const r = await ask("ui/initialize", { protocolVersion: "2026-01-26", appInfo: { name: "fake-chart", version: "1" },
    appCapabilities: {} });
  theme(((r.result || {}).hostContext) || {});
  send({ method: "ui/notifications/initialized", params: {} });
  new ResizeObserver(() => send({ method: "ui/notifications/size-changed",
    params: { width: document.body.scrollWidth, height: document.body.scrollHeight } })).observe(document.body);
})();
</script></body></html>
"""


def text_result(text: str, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "echo":
        return text_result(f"echo: {args.get('text', '')}")
    if name == "add":
        return text_result(str(args.get("a", 0) + args.get("b", 0)))
    if name == "write_note":
        NOTES.append(str(args.get("text", "")))
        return text_result(f"saved note {len(NOTES)}")
    if name == "fail":
        return text_result("boom", True)
    if name == "crash":
        os._exit(3)
    if name == "show_chart":
        out = text_result("chart of 3 values")
        out["structuredContent"] = {"title": str(args.get("title") or "Chart"), "values": [3, 5, 8]}
        return out
    if name == "app_bump":
        BUMPS[0] += 1
        out = text_result(f"bumped {BUMPS[0]}")
        out["structuredContent"] = {"count": BUMPS[0]}
        return out
    raise KeyError(name)


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """The response to a request, or None for a notification."""
    method, ident = message.get("method"), message.get("id")
    if ident is None:
        return None
    params = message.get("params") or {}
    if method == "initialize":
        result: dict[str, Any] = {
            "protocolVersion": PROTOCOL,
            "capabilities": {"tools": {}, "resources": {}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0"},
        }
    elif method == "tools/list":
        page = int(os.environ.get("FAKE_MCP_PAGE") or 0)
        if page:
            start = int(params.get("cursor") or 0)
            chunk = TOOLS[start : start + page]
            result = {"tools": chunk}
            if start + page < len(TOOLS):
                result["nextCursor"] = str(start + page)
        else:
            result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            result = call_tool(str(params.get("name")), params.get("arguments") or {})
        except KeyError:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32602, "message": "Unknown tool"}}
    elif method == "resources/read":
        if params.get("uri") != APP_URI:
            return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32002, "message": "Resource not found"}}
        result = {
            "contents": [
                {
                    "uri": APP_URI,
                    "mimeType": APP_MIME,
                    "text": APP_HTML,
                    "_meta": {"ui": {"prefersBorder": True, "csp": {"connectDomains": [], "resourceDomains": []}}},
                }
            ]
        }
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": ident, "error": {"code": -32601, "message": f"No method {method}"}}
    return {"jsonrpc": "2.0", "id": ident, "result": result}


def serve_stdio() -> int:
    for stream in (sys.stdout, sys.stdin):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    sys.stderr.write("fake mcp ready\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


# ------------------------------------------------------------------------------------------------ OAuth stand-in


class Auth:
    """What the OAuth stand-in remembers. One instance per process."""

    def __init__(self) -> None:
        self.enabled = False
        self.dcr = True
        self.lock = threading.Lock()
        self.clients: dict[str, dict[str, Any]] = {}
        self.codes: dict[str, dict[str, Any]] = {}
        self.access: dict[str, float] = {}  # token -> expiry
        self.refresh: dict[str, str] = {}  # refresh token -> client id
        self.log: list[dict[str, Any]] = []
        self.counter = 0

    def note(self, **entry: Any) -> None:
        with self.lock:
            self.log.append(entry)

    def new(self, prefix: str) -> str:
        with self.lock:
            self.counter += 1
            return f"{prefix}-{self.counter}-{secrets.token_hex(6)}"


AUTH = Auth()
SCOPE = "mcp:tools"


def s256(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


class HttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    sse = False
    legacy = False
    session = "fake-session-1"
    streams: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def reply(
        self, status: int, body: bytes, ctype: str = "application/json", extra: dict[str, str] | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, status: int, doc: Any, extra: dict[str, str] | None = None) -> None:
        self.reply(status, json.dumps(doc).encode(), "application/json", extra)

    def authorized(self) -> bool:
        if AUTH.enabled:
            header = self.headers.get("Authorization") or ""
            token = header[7:] if header.startswith("Bearer ") else ""
            with AUTH.lock:
                expiry = AUTH.access.get(token)
            good = bool(expiry and expiry > time.time())
            AUTH.note(event="mcp-request", ok=good, token_in_query="access_token" in self.path)
            return bool(expiry and expiry > time.time())
        token = os.environ.get("FAKE_MCP_TOKEN")
        return not token or self.headers.get("Authorization") == f"Bearer {token}"

    def unauthorized(self) -> None:
        challenge = f'Bearer resource_metadata="{self.base}/.well-known/oauth-protected-resource/mcp", scope="{SCOPE}"'
        if not AUTH.enabled:
            challenge = "Bearer"
        self.reply(401, b'{"error": "bad token"}', "application/json", {"WWW-Authenticate": challenge})

    def form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        return {k: v[-1] for k, v in parse_qs(self.rfile.read(length).decode()).items()}

    # ---- OAuth endpoints

    def oauth_get(self, u: Any) -> bool:
        path = u.path
        if path == "/.well-known/oauth-protected-resource/mcp":
            self.json(
                200,
                {
                    "resource": f"{self.base}/mcp",
                    "authorization_servers": [self.base],
                    "scopes_supported": [SCOPE],
                },
            )
            return True
        if path == "/.well-known/oauth-authorization-server":
            doc: dict[str, Any] = {
                "issuer": self.base,
                "authorization_endpoint": f"{self.base}/authorize",
                "token_endpoint": f"{self.base}/token",
                "revocation_endpoint": f"{self.base}/revoke",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": ["none"],
            }
            if AUTH.dcr:
                doc["registration_endpoint"] = f"{self.base}/register"
            self.json(200, doc)
            return True
        if path == "/authorize":
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            ok = (
                q.get("response_type") == "code"
                and q.get("code_challenge_method") == "S256"
                and q.get("code_challenge")
                and q.get("resource") == f"{self.base}/mcp"
                and (q.get("client_id") in AUTH.clients or not AUTH.dcr)
                and q.get("redirect_uri", "").startswith("http://127.0.0.1:")
            )
            AUTH.note(event="authorize", ok=bool(ok), resource=q.get("resource"), scope=q.get("scope"))
            if not ok:
                self.reply(400, b"bad authorization request", "text/plain")
                return True
            code = AUTH.new("code")
            with AUTH.lock:
                AUTH.codes[code] = {
                    "challenge": q["code_challenge"],
                    "redirect": q["redirect_uri"],
                    "client": q.get("client_id"),
                }
            sep = "&" if "?" in q["redirect_uri"] else "?"
            back = q["redirect_uri"] + sep + urlencode({"code": code, "state": q.get("state", "")})
            self.reply(302, b"", "text/plain", {"Location": back})
            return True
        if path == "/_state":
            self.json(200, {"log": AUTH.log, "notes": NOTES, "bumps": BUMPS[0], "clients": len(AUTH.clients)})
            return True
        return False

    def oauth_post(self, u: Any) -> bool:
        path = u.path
        if path == "/register" and AUTH.dcr:
            length = int(self.headers.get("Content-Length") or 0)
            doc = json.loads(self.rfile.read(length) or b"{}")
            client = AUTH.new("client")
            AUTH.clients[client] = doc
            AUTH.note(
                event="register", method=doc.get("token_endpoint_auth_method"), redirects=doc.get("redirect_uris")
            )
            self.json(201, {"client_id": client, "redirect_uris": doc.get("redirect_uris", [])})
            return True
        if path == "/token":
            f = self.form()
            expires = int(os.environ.get("FAKE_MCP_EXPIRES") or 3600)
            if f.get("grant_type") == "authorization_code":
                with AUTH.lock:
                    rec = AUTH.codes.pop(f.get("code", ""), None)
                ok = bool(
                    rec
                    and s256(f.get("code_verifier", "")) == rec["challenge"]
                    and f.get("redirect_uri") == rec["redirect"]
                    and f.get("resource") == f"{self.base}/mcp"
                )
                AUTH.note(event="token", grant="authorization_code", ok=ok, resource=f.get("resource"))
                if not ok:
                    self.json(400, {"error": "invalid_grant"})
                    return True
            elif f.get("grant_type") == "refresh_token":
                with AUTH.lock:
                    known = AUTH.refresh.pop(f.get("refresh_token", ""), None)
                AUTH.note(event="token", grant="refresh_token", ok=bool(known), resource=f.get("resource"))
                if known is None or f.get("resource") != f"{self.base}/mcp":
                    self.json(400, {"error": "invalid_grant"})
                    return True
            else:
                self.json(400, {"error": "unsupported_grant_type"})
                return True
            access, refresh = AUTH.new("at"), AUTH.new("rt")
            with AUTH.lock:
                AUTH.access[access] = time.time() + expires
                AUTH.refresh[refresh] = f.get("client_id", "")
            self.json(
                200,
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "expires_in": expires,
                    "refresh_token": refresh,
                    "scope": SCOPE,
                },
            )
            return True
        if path == "/revoke":
            f = self.form()
            with AUTH.lock:
                AUTH.access.pop(f.get("token", ""), None)
                AUTH.refresh.pop(f.get("token", ""), None)
            AUTH.note(event="revoke", hint=f.get("token_type_hint"))
            self.json(200, {})
            return True
        return False

    # ---- MCP

    def do_DELETE(self) -> None:
        self.reply(200, b"{}")

    def do_GET(self) -> None:
        u = urlparse(self.path)
        if self.oauth_get(u):
            return
        if self.legacy and u.path == "/sse":
            sid = secrets.token_hex(4)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            queue: list[str] = []
            HttpHandler.streams[sid] = queue
            self.close_connection = True
            try:
                self.wfile.write(f"event: endpoint\ndata: /messages?session={sid}\n\n".encode())
                self.wfile.flush()
                while True:
                    while queue:
                        self.wfile.write(f"event: message\ndata: {queue.pop(0)}\n\n".encode())
                        self.wfile.flush()
                    time.sleep(0.02)
            except OSError:
                HttpHandler.streams.pop(sid, None)
            return
        self.reply(404, b"{}")

    def do_POST(self) -> None:
        u = urlparse(self.path)
        if self.oauth_post(u):
            return
        length = int(self.headers.get("Content-Length") or 0)
        message = json.loads(self.rfile.read(length) or b"{}")
        if self.legacy:
            if u.path == "/messages":
                queue = HttpHandler.streams.get(parse_qs(u.query).get("session", [""])[0])
                if queue is None:
                    self.reply(404, b"{}")
                    return
                response = handle(message)
                if response is not None:
                    queue.append(json.dumps(response))
                self.reply(202, b"Accepted", "text/plain")
                return
            self.reply(405, b"Method Not Allowed", "text/plain")  # the old transport has no POST on its URL
            return
        if u.path != "/mcp":
            self.reply(404, b"{}")
            return
        if not self.authorized():
            self.unauthorized()
            return
        extra = {"Mcp-Session-Id": self.session}
        if message.get("method") != "initialize" and self.headers.get("Mcp-Session-Id") != self.session:
            self.reply(400, b'{"error": "missing session"}')
            return
        response = handle(message)
        if response is None:
            self.reply(202, b"", "text/plain", extra)
        elif self.sse and message.get("method") == "tools/call":
            ping = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "hi"}}
            body = (f"data: {json.dumps(ping)}\n\ndata: {json.dumps(response)}\n\n").encode()
            self.reply(200, body, "text/event-stream", extra)
        else:
            self.reply(200, json.dumps(response).encode(), "application/json", extra)


def serve_http(port: int, sse: bool, oauth: bool = False, legacy: bool = False, dcr: bool = True) -> int:
    HttpHandler.sse = sse
    HttpHandler.legacy = legacy
    AUTH.enabled, AUTH.dcr = oauth, dcr
    httpd = ThreadingHTTPServer(("127.0.0.1", port), HttpHandler)
    httpd.daemon_threads = True
    sys.stderr.write(f"fake mcp listening on {port}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str]) -> int:
    if "--http" in argv:
        return serve_http(
            int(argv[argv.index("--http") + 1]),
            "--sse" in argv,
            oauth="--oauth" in argv,
            legacy="--legacy" in argv,
            dcr="--no-dcr" not in argv,
        )
    return serve_stdio()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
