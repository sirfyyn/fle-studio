"""A small MCP client (Model Context Protocol), standard library only.

Two kinds of servers are supported, written in the usual `mcpServers` JSON:

    {"mcpServers": {
        "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/folder"]},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer …"}}
    }}

* stdio: a program that FLE Studio starts and talks to over its standard input and output (JSON-RPC, one message
  per line).
* Streamable HTTP: a URL that takes JSON-RPC in POST requests and answers with JSON or an event stream.

The old HTTP+SSE transport is not supported. Only tools are used (no prompts, resources or sampling).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urlparse

from . import __version__, net
from .paths import IS_WIN
from .procutil import KillOnCloseJob

PROTOCOL = "2025-06-18"
CONNECT_TIMEOUT = 90.0  # a first `npx -y …` may have to download the server
MAX_RESULT_CHARS = 60_000


class McpError(RuntimeError):
    """A server could not be reached or answered with an error."""


# ------------------------------------------------------------------------------------------------ configuration


def parse_servers(text: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """(servers, problems) from the `mcpServers` JSON text. A server with a problem is left out and named in
    `problems`; an unreadable text gives the problem under the empty name."""
    text = (text or "").strip()
    if not text:
        return {}, {}
    try:
        doc = json.loads(text)
    except ValueError as exc:
        return {}, {"": f"The MCP servers text is not valid JSON: {exc}"}
    if isinstance(doc, dict) and isinstance(doc.get("mcpServers"), dict):
        doc = doc["mcpServers"]
    if not isinstance(doc, dict):
        return {}, {"": 'Expected {"mcpServers": {"name": {"command": …} or {"url": …}}}'}
    servers: dict[str, dict[str, Any]] = {}
    problems: dict[str, str] = {}
    for name, entry in doc.items():
        name = str(name)
        if not isinstance(entry, dict):
            problems[name] = "The entry must be an object"
            continue
        if entry.get("disabled") is True:
            continue
        kind = str(entry.get("type") or "").lower()
        if kind == "sse":
            problems[name] = "The old HTTP+SSE transport is not supported; use a Streamable HTTP URL or a command"
        elif isinstance(entry.get("command"), str) and entry["command"].strip():
            args = entry.get("args") or []
            env = entry.get("env") or {}
            if not isinstance(args, list) or not isinstance(env, dict):
                problems[name] = '"args" must be a list and "env" an object'
                continue
            servers[name] = {
                "transport": "stdio",
                "command": entry["command"].strip(),
                "args": [str(a) for a in args],
                "env": {str(k): str(v) for k, v in env.items()},
                "cwd": str(entry["cwd"]) if entry.get("cwd") else "",
            }
        elif isinstance(entry.get("url"), str) and entry["url"].strip():
            url = entry["url"].strip()
            headers = entry.get("headers") or {}
            if urlparse(url).scheme not in ("http", "https") or not urlparse(url).hostname:
                problems[name] = "The URL must start with http:// or https://"
            elif not isinstance(headers, dict):
                problems[name] = '"headers" must be an object'
            else:
                servers[name] = {
                    "transport": "http",
                    "url": url,
                    "headers": {str(k): str(v) for k, v in headers.items()},
                }
        else:
            problems[name] = 'Needs a "command" (a program to start) or a "url" (a Streamable HTTP server)'
    return servers, problems


# ------------------------------------------------------------------------------------------------ transports


_job: KillOnCloseJob | None = None


def _windows_job() -> KillOnCloseJob:
    global _job
    if _job is None:
        _job = KillOnCloseJob()
    return _job


class _Waiter:
    __slots__ = ("event", "message")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.message: dict[str, Any] | None = None


class StdioTransport:
    """JSON-RPC over a child process's standard input and output."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.proc: subprocess.Popen[bytes] | None = None
        self.stderr_tail: deque[str] = deque(maxlen=30)
        self._pending: dict[int, _Waiter] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._closed = ""

    def start(self) -> None:
        env = {**os.environ, **self.spec["env"]}
        command = shutil.which(self.spec["command"], path=env.get("PATH")) or ""
        if not command:
            raise McpError(f"The command '{self.spec['command']}' was not found. Is it installed and on the PATH?")
        kwargs: dict[str, Any] = {}
        if IS_WIN:  # no console window; `npx` and friends are .cmd files that `which` resolved above
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            self.proc = subprocess.Popen(
                [command, *self.spec["args"]],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.spec["cwd"] or None,
                **kwargs,
            )
        except OSError as exc:
            raise McpError(f"Could not start '{self.spec['command']}': {exc}") from None
        if IS_WIN:  # a hard kill of FLE Studio (no atexit) must not leave the server behind
            with contextlib.suppress(Exception):
                _windows_job().add(self.proc)
        threading.Thread(target=self._read_loop, name="mcp-stdout", daemon=True).start()
        threading.Thread(target=self._stderr_loop, name="mcp-stderr", daemon=True).start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and not self._closed

    def _tail(self) -> str:
        return " | ".join(list(self.stderr_tail)[-4:])

    def _stderr_loop(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self.stderr_tail.append(line[:300])

    def _read_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self.stderr_tail.append(f"(not JSON on stdout) {line[:200]}")
                    continue
                if isinstance(message, dict):
                    self._dispatch(message)
        except (OSError, ValueError):
            pass
        finally:
            tail = self._tail()
            self._fail_all("The server exited" + (f": {tail}" if tail else ""))

    def _fail_all(self, why: str) -> None:
        self._closed = self._closed or why
        with self._lock:
            waiters, self._pending = list(self._pending.values()), {}
        for w in waiters:
            w.message = {"error": {"code": -32000, "message": why}}
            w.event.set()

    def _dispatch(self, message: dict[str, Any]) -> None:
        ident = message.get("id")
        if "method" in message and ident is not None:  # a request from the server
            if message["method"] == "ping":
                self._send({"jsonrpc": "2.0", "id": ident, "result": {}})
            else:
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": ident,
                        "error": {"code": -32601, "message": "FLE Studio does not offer that"},
                    }
                )
            return
        if "method" in message:
            return  # a notification (progress, list changes): not needed
        with self._lock:
            waiter = self._pending.pop(ident, None) if isinstance(ident, int) else None
        if waiter is not None:
            waiter.message = message
            waiter.event.set()

    def _send(self, message: dict[str, Any]) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise McpError("The server is not running" + (f": {self._tail()}" if self._tail() else ""))
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                proc.stdin.write(data)
                proc.stdin.flush()
        except (OSError, ValueError):
            raise McpError("The server closed its input" + (f": {self._tail()}" if self._tail() else "")) from None

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        waiter = _Waiter()
        with self._lock:
            self._next += 1
            ident = self._next
            self._pending[ident] = waiter
        try:
            self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}})
            if not waiter.event.wait(timeout):
                raise McpError(f"No answer to {method} within {timeout:g} s")
        finally:
            with self._lock:
                self._pending.pop(ident, None)
        reply = waiter.message or {}
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {}
            raise McpError(str(err.get("message") or reply["error"]))
        return reply.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        proc = self.proc
        self._closed = self._closed or "closed"
        if proc is None:
            return
        with contextlib.suppress(OSError, ValueError):
            if proc.stdin:
                proc.stdin.close()  # most servers quit when their input ends
        try:
            proc.wait(timeout=1.5)
            return
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        if IS_WIN:  # terminate() ends only the .cmd shim, not the program behind it
            with contextlib.suppress(OSError, subprocess.SubprocessError):
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                )
        with contextlib.suppress(OSError):
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2)


class HttpTransport:
    """JSON-RPC in POST requests (MCP "Streamable HTTP"). One connection per request keeps this simple."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.session = ""
        self.version = ""
        self._next = 0
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return True

    def start(self) -> None:
        return None

    def _connection(self, timeout: float) -> Any:
        import http.client  # noqa: PLC0415

        u = urlparse(self.spec["url"])
        if u.scheme == "https":
            return http.client.HTTPSConnection(
                u.hostname or "", u.port or 443, timeout=timeout, context=net.ssl_context()
            )
        return http.client.HTTPConnection(u.hostname or "", u.port or 80, timeout=timeout)

    def _post(self, payload: dict[str, Any], timeout: float, want: int | None) -> dict[str, Any] | None:
        u = urlparse(self.spec["url"])
        target = (u.path or "/") + (f"?{u.query}" if u.query else "")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": net.USER_AGENT,
            **self.spec["headers"],
        }
        if self.session:
            headers["Mcp-Session-Id"] = self.session
        if self.version:
            headers["MCP-Protocol-Version"] = self.version
        conn = self._connection(timeout)
        deadline = time.monotonic() + timeout
        try:
            try:
                conn.request("POST", target, body=json.dumps(payload).encode("utf-8"), headers=headers)
                resp = conn.getresponse()
            except OSError as exc:
                raise McpError(f"Could not reach {u.hostname}: {exc}") from None
            self.session = resp.getheader("Mcp-Session-Id") or self.session
            if resp.status >= 400:
                body = resp.read(2000).decode("utf-8", "replace").strip()
                raise McpError(f"HTTP {resp.status} from the server" + (f": {body[:300]}" if body else ""))
            if want is None:
                resp.read()
                return None
            ctype = (resp.getheader("Content-Type") or "").lower()
            if "text/event-stream" in ctype:
                return self._read_events(resp, want, deadline)
            try:
                doc = json.loads(resp.read().decode("utf-8"))
            except (ValueError, OSError) as exc:
                raise McpError(f"The server answered with something that is not JSON ({exc})") from None
            for item in doc if isinstance(doc, list) else [doc]:
                if isinstance(item, dict) and item.get("id") == want:
                    return item
            raise McpError("The server's answer did not match the request")
        except OSError as exc:  # also a read timeout
            raise McpError(f"No answer within {timeout:g} s" if isinstance(exc, TimeoutError) else str(exc)) from None
        finally:
            conn.close()

    @staticmethod
    def _read_events(resp: Any, want: int, deadline: float) -> dict[str, Any]:
        data: list[str] = []
        while time.monotonic() < deadline:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
            elif not line and data:
                text, data = "\n".join(data), []
                try:
                    item = json.loads(text)
                except ValueError:
                    continue
                for one in item if isinstance(item, list) else [item]:
                    if isinstance(one, dict) and one.get("id") == want and ("result" in one or "error" in one):
                        return one
        raise McpError("The event stream ended without an answer")

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        with self._lock:
            self._next += 1
            ident = self._next
        reply = self._post({"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}, timeout, ident)
        reply = reply or {}
        if "error" in reply:
            err = reply["error"] if isinstance(reply["error"], dict) else {}
            raise McpError(str(err.get("message") or reply["error"]))
        return reply.get("result")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params or {}}, 15.0, None)

    def close(self) -> None:
        if not self.session:
            return
        with contextlib.suppress(Exception):
            conn = self._connection(5.0)
            u = urlparse(self.spec["url"])
            conn.request(
                "DELETE",
                (u.path or "/") + (f"?{u.query}" if u.query else ""),
                headers={"Mcp-Session-Id": self.session, **self.spec["headers"]},
            )
            conn.getresponse().read()
            conn.close()
        self.session = ""


# ------------------------------------------------------------------------------------------------ one server


def result_text(result: Any) -> tuple[str, bool]:
    """(text, is_error) from a `tools/call` result. Pictures and other media are named, not included."""
    if not isinstance(result, dict):
        return str(result or ""), False
    parts: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text") or ""))
        elif kind == "image":
            parts.append(f"[image {item.get('mimeType', '')} not shown]")
        elif kind == "audio":
            parts.append(f"[audio {item.get('mimeType', '')} not shown]")
        elif kind == "resource_link":
            parts.append(f"[resource {item.get('uri', '')}]")
        elif kind == "resource":
            res = item.get("resource") or {}
            parts.append(
                str(res.get("text")) if isinstance(res.get("text"), str) else f"[resource {res.get('uri', '')}]"
            )
    text = "\n".join(p for p in parts if p)
    if not text and result.get("structuredContent") is not None:
        text = json.dumps(result["structuredContent"], ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n[cut off after {MAX_RESULT_CHARS} characters]"
    return text, bool(result.get("isError"))


class McpServer:
    def __init__(self, name: str, spec: dict[str, Any], call_timeout: float = 60.0) -> None:
        self.name = name
        self.spec = spec
        self.call_timeout = call_timeout
        self.status = "idle"  # idle | connecting | ready | error
        self.error = ""
        self.tools: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.transport: StdioTransport | HttpTransport | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Start the server and read its tool list; sets status and error instead of raising."""
        with self._lock:
            self.status, self.error = "connecting", ""
            transport: StdioTransport | HttpTransport = (
                StdioTransport(self.spec) if self.spec["transport"] == "stdio" else HttpTransport(self.spec)
            )
            self.transport = transport
            try:
                transport.start()
                result = transport.request(
                    "initialize",
                    {
                        "protocolVersion": PROTOCOL,
                        "capabilities": {},
                        "clientInfo": {"name": "fle-studio", "version": __version__},
                    },
                    CONNECT_TIMEOUT,
                )
                result = result if isinstance(result, dict) else {}
                self.info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
                if isinstance(transport, HttpTransport):
                    transport.version = str(result.get("protocolVersion") or PROTOCOL)
                transport.notify("notifications/initialized")
                self.tools = self._list_tools(transport)
                self.status = "ready"
            except (McpError, OSError, ValueError) as exc:
                self.status, self.error, self.tools = "error", str(exc), []
                transport.close()

    def _list_tools(self, transport: StdioTransport | HttpTransport) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor = ""
        for _ in range(20):
            result = transport.request("tools/list", {"cursor": cursor} if cursor else {}, 30.0)
            result = result if isinstance(result, dict) else {}
            tools += [t for t in result.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("name"), str)]
            cursor = str(result.get("nextCursor") or "")
            if not cursor:
                break
        return tools

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; (text, is_error). Raises McpError when the server cannot be reached."""
        transport = self.transport
        if self.status != "ready" or transport is None:
            raise McpError(f"The server '{self.name}' is not connected" + (f": {self.error}" if self.error else ""))
        if not transport.alive():  # it exited since the last call: one new start, before anything was sent
            self.connect()
            transport = self.transport
            if self.status != "ready" or transport is None:
                raise McpError(f"The server '{self.name}' stopped and did not start again: {self.error}")
        result = transport.request("tools/call", {"name": tool, "arguments": arguments}, self.call_timeout)
        return result_text(result)

    def close(self) -> None:
        if self.transport is not None:
            with contextlib.suppress(Exception):
                self.transport.close()
        self.status = "idle"


# ------------------------------------------------------------------------------------------------ all servers


class McpManager:
    """The servers named in the settings. They connect in the background and stay connected until the settings
    change or FLE Studio ends."""

    def __init__(self) -> None:
        self._servers: dict[str, McpServer] = {}
        self._lock = threading.RLock()
        atexit.register(self.close_all)

    def sync(self, specs: dict[str, dict[str, Any]], call_timeout: float = 60.0) -> None:
        with self._lock:
            for name in list(self._servers):
                if name not in specs or self._servers[name].spec != specs[name]:
                    server = self._servers.pop(name)
                    threading.Thread(target=server.close, daemon=True).start()
            for name, spec in specs.items():
                server = self._servers.get(name)
                if server is None:
                    server = self._servers[name] = McpServer(name, spec, call_timeout)
                    threading.Thread(target=server.connect, name=f"mcp-{name}", daemon=True).start()
                else:
                    server.call_timeout = call_timeout

    def reconnect(self, name: str | None = None) -> None:
        with self._lock:
            targets = [s for n, s in self._servers.items() if name in (None, n)]
        for server in targets:
            server.status = "connecting"  # at once, so a listing right after this call does not show the old state
            threading.Thread(target=lambda s=server: (s.close(), s.connect()), daemon=True).start()

    def get(self, name: str) -> McpServer | None:
        with self._lock:
            return self._servers.get(name)

    def servers(self) -> list[McpServer]:
        with self._lock:
            return list(self._servers.values())

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no server is still connecting (for tests and the Settings "test" button)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if not any(s.status in ("idle", "connecting") for s in self.servers()):
                return True
            time.sleep(0.05)
        return False

    def close_all(self) -> None:
        with self._lock:
            servers, self._servers = list(self._servers.values()), {}
        for server in servers:
            with contextlib.suppress(Exception):
                server.close()
