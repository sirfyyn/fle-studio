#!/usr/bin/env python3
"""A tiny MCP server for tests (standard library only; macOS, Linux and Windows).

    python fake_mcp_server.py                     speak MCP over stdin/stdout (one JSON message per line)
    python fake_mcp_server.py --http PORT         Streamable HTTP on 127.0.0.1:PORT/mcp (JSON answers)
    python fake_mcp_server.py --http PORT --sse   the same, but tools/call answers as an event stream

Tools: echo, add (read-only hint), write_note (changes state), fail (isError result), crash (exits without answering).
Environment: FAKE_MCP_PAGE=N lists N tools per page (to test paging); FAKE_MCP_TOKEN=T makes the HTTP mode demand
``Authorization: Bearer T``.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROTOCOL = "2025-06-18"
NOTES: list[str] = []

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
]


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
            "capabilities": {"tools": {}},
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


class HttpHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    sse = False
    session = "fake-session-1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

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

    def authorized(self) -> bool:
        token = os.environ.get("FAKE_MCP_TOKEN")
        return not token or self.headers.get("Authorization") == f"Bearer {token}"

    def do_DELETE(self) -> None:
        self.reply(200, b"{}")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        message = json.loads(self.rfile.read(length) or b"{}")
        if self.path.split("?")[0] != "/mcp":
            self.reply(404, b"{}")
            return
        if not self.authorized():
            self.reply(401, b'{"error": "bad token"}')
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


def serve_http(port: int, sse: bool) -> int:
    HttpHandler.sse = sse
    httpd = ThreadingHTTPServer(("127.0.0.1", port), HttpHandler)
    sys.stderr.write(f"fake mcp listening on {port}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str]) -> int:
    if "--http" in argv:
        return serve_http(int(argv[argv.index("--http") + 1]), "--sse" in argv)
    return serve_stdio()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
