"""A small MCP client (Model Context Protocol), standard library only.

Two kinds of servers are supported, written in the usual `mcpServers` JSON:

    {"mcpServers": {
        "files":  {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/some/folder"]},
        "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": "Bearer …"}}
    }}

* stdio: a program that FLE Studio starts and talks to over its standard input and output (JSON-RPC, one message
  per line).
* Streamable HTTP: a URL that takes JSON-RPC in POST requests and answers with JSON or an event stream.

A Streamable HTTP server that wants a sign-in answers 401; FLE Studio then offers a "Sign in" button (OAuth, see
oauth.py). A server of the older 2024-11-05 revision (a GET that opens an event stream and names the POST address in
its first event) is used when the POST to the URL fails with a 4xx, or when the entry says ``"type": "sse"``.
Tools are used, and the ``ui://`` resources that MCP Apps point at; prompts and sampling are not.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Any
from urllib.parse import urljoin, urlparse

from . import __version__, net
from .oauth import OAuthManager, Provider, bearer_challenge
from .paths import IS_WIN
from .procutil import KillOnCloseJob

PROTOCOL = "2025-11-25"
CONNECT_TIMEOUT = 90.0  # a first `npx -y …` may have to download the server
MAX_RESULT_CHARS = 60_000
UI_EXTENSION = "io.modelcontextprotocol/ui"
UI_MIME = "text/html;profile=mcp-app"
MAX_APP_BYTES = 4_000_000


class McpError(RuntimeError):
    """A server could not be reached or answered with an error."""


class McpHttpError(McpError):
    """The server answered a request with an HTTP error status."""

    def __init__(self, status: int, text: str, www_authenticate: str = "") -> None:
        super().__init__(text)
        self.status = status
        self.www_authenticate = www_authenticate


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
        if isinstance(entry.get("command"), str) and entry["command"].strip():
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
            elif entry.get("oauth") is not None and not isinstance(entry["oauth"], dict):
                problems[name] = '"oauth" must be an object like {"client_id": "…", "scope": "…"}'
            else:
                spec: dict[str, Any] = {
                    "transport": "sse" if kind == "sse" else "http",
                    "url": url,
                    "headers": {str(k): str(v) for k, v in headers.items()},
                }
                oauth = entry.get("oauth") or {}
                if oauth:
                    spec["oauth"] = {
                        k: str(oauth[k]) for k in ("client_id", "client_secret", "scope") if oauth.get(k)
                    }
                servers[name] = spec
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

    def __init__(self, spec: dict[str, Any], provider: Provider | None = None) -> None:
        self.spec = spec
        self.provider = provider  # supplies the OAuth token, when there is one
        self.session = ""
        self.version = ""
        self._next = 0
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return True

    def start(self) -> None:
        return None

    def _auth(self) -> dict[str, str]:
        """The Authorization header from the sign-in, unless the entry sets its own."""
        if self.provider is None or any(k.lower() == "authorization" for k in self.spec["headers"]):
            return {}
        value = self.provider.header()
        return {"Authorization": value} if value else {}

    def _connection(self, timeout: float) -> Any:
        import http.client  # noqa: PLC0415

        u = urlparse(self.spec["url"])
        if u.scheme == "https":
            return http.client.HTTPSConnection(
                u.hostname or "", u.port or 443, timeout=timeout, context=net.ssl_context()
            )
        return http.client.HTTPConnection(u.hostname or "", u.port or 80, timeout=timeout)

    def _post(self, payload: dict[str, Any], timeout: float, want: int | None) -> dict[str, Any] | None:
        try:
            return self._post_once(payload, timeout, want)
        except McpHttpError as exc:
            # A 401 with a refresh token to hand: get a new token and repeat the request once.
            if exc.status == 401 and self.provider is not None and self.provider.refresh():
                return self._post_once(payload, timeout, want)
            raise

    def _post_once(self, payload: dict[str, Any], timeout: float, want: int | None) -> dict[str, Any] | None:
        u = urlparse(self.spec["url"])
        target = (u.path or "/") + (f"?{u.query}" if u.query else "")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": net.USER_AGENT,
            **self.spec["headers"],
            **self._auth(),
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
                raise McpHttpError(
                    resp.status,
                    f"HTTP {resp.status} from the server" + (f": {body[:300]}" if body else ""),
                    resp.getheader("WWW-Authenticate") or "",
                )
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
                headers={"Mcp-Session-Id": self.session, **self.spec["headers"], **self._auth()},
            )
            conn.getresponse().read()
            conn.close()
        self.session = ""


class SseTransport:
    """The older HTTP+SSE transport (revision 2024-11-05): a GET that stays open and delivers the server's messages as
    events, and a separate address (named by the first event, ``endpoint``) that takes the client's messages in
    POSTs."""

    def __init__(self, spec: dict[str, Any], provider: Provider | None = None) -> None:
        self.spec = spec
        self.provider = provider
        self.version = ""
        self.session = ""
        self._pending: dict[int, _Waiter] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._endpoint = ""
        self._ready = threading.Event()
        self._failure: McpError | None = None
        self._closed = ""
        self._stream: Any = None

    def alive(self) -> bool:
        return not self._closed and self._failure is None

    def _headers(self) -> dict[str, str]:
        extra: dict[str, str] = {}
        if self.provider is not None and not any(k.lower() == "authorization" for k in self.spec["headers"]):
            value = self.provider.header()
            if value:
                extra["Authorization"] = value
        return {"User-Agent": net.USER_AGENT, **self.spec["headers"], **extra}

    def _connection(self, url: str, timeout: float | None) -> Any:
        import http.client  # noqa: PLC0415

        u = urlparse(url)
        if u.scheme == "https":
            return http.client.HTTPSConnection(
                u.hostname or "", u.port or 443, timeout=timeout, context=net.ssl_context()
            )
        return http.client.HTTPConnection(u.hostname or "", u.port or 80, timeout=timeout)

    def start(self) -> None:
        threading.Thread(target=self._read_loop, name="mcp-sse", daemon=True).start()
        if not self._ready.wait(30.0):
            self.close()
            raise McpError("The server did not name its message address (no `endpoint` event within 30 s)")
        if self._failure is not None:
            raise self._failure

    def _read_loop(self) -> None:
        url = self.spec["url"]
        u = urlparse(url)
        conn = self._connection(url, 30.0)
        try:
            try:
                conn.request(
                    "GET",
                    (u.path or "/") + (f"?{u.query}" if u.query else ""),
                    headers={"Accept": "text/event-stream", **self._headers()},
                )
                resp = conn.getresponse()
            except OSError as exc:
                self._failure = McpError(f"Could not reach {u.hostname}: {exc}")
                return
            if resp.status >= 400:
                body = resp.read(1000).decode("utf-8", "replace").strip()
                self._failure = McpHttpError(
                    resp.status, f"HTTP {resp.status} from the server" + (f": {body[:300]}" if body else ""),
                    resp.getheader("WWW-Authenticate") or "",
                )
                return
            self._stream = conn
            if conn.sock is not None:
                conn.sock.settimeout(None)
            event, data = "message", []
            while not self._closed:
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip() or "message"
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
                elif not line:
                    if data:
                        self._event(event, "\n".join(data), url)
                    event, data = "message", []
        except (OSError, ValueError):
            pass
        finally:
            with contextlib.suppress(Exception):
                conn.close()
            if not self._closed:
                self._failure = self._failure or McpError("The server closed the event stream")
            self._fail_all(str(self._failure or "closed"))
            self._ready.set()

    def _event(self, event: str, text: str, base: str) -> None:
        if event == "endpoint":
            target = urljoin(base, text.strip())
            a, b = urlparse(target), urlparse(base)
            if (a.scheme, a.netloc) != (b.scheme, b.netloc):  # never send to another host than the one configured
                self._failure = McpError("The server named a message address on a different host; refusing it")
            else:
                self._endpoint = target
            self._ready.set()
            return
        try:
            message = json.loads(text)
        except ValueError:
            return
        for one in message if isinstance(message, list) else [message]:
            if isinstance(one, dict) and "method" not in one:
                with self._lock:
                    waiter = self._pending.pop(one.get("id"), None) if isinstance(one.get("id"), int) else None
                if waiter is not None:
                    waiter.message = one
                    waiter.event.set()

    def _fail_all(self, why: str) -> None:
        with self._lock:
            waiters, self._pending = list(self._pending.values()), {}
        for w in waiters:
            w.message = {"error": {"code": -32000, "message": why}}
            w.event.set()

    def _send(self, payload: dict[str, Any], timeout: float) -> None:
        if not self._endpoint:
            raise self._failure or McpError("Not connected")
        conn = self._connection(self._endpoint, timeout)
        u = urlparse(self._endpoint)
        try:
            conn.request(
                "POST",
                (u.path or "/") + (f"?{u.query}" if u.query else ""),
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", **self._headers()},
            )
            resp = conn.getresponse()
            body = resp.read(2000).decode("utf-8", "replace").strip()
            if resp.status >= 400:
                raise McpHttpError(
                    resp.status, f"HTTP {resp.status} from the server" + (f": {body[:300]}" if body else ""),
                    resp.getheader("WWW-Authenticate") or "",
                )
        except OSError as exc:
            raise McpError(f"Could not reach {u.hostname}: {exc}") from None
        finally:
            conn.close()

    def request(self, method: str, params: dict[str, Any] | None, timeout: float) -> Any:
        waiter = _Waiter()
        with self._lock:
            self._next += 1
            ident = self._next
            self._pending[ident] = waiter
        try:
            self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params or {}}, min(timeout, 30.0))
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
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}}, 15.0)

    def close(self) -> None:
        self._closed = self._closed or "closed"
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                if stream.sock is not None:
                    stream.sock.shutdown(2)
                stream.close()
        self._ready.set()


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
    def __init__(
        self, name: str, spec: dict[str, Any], call_timeout: float = 60.0, oauth: OAuthManager | None = None
    ) -> None:
        self.name = name
        self.spec = spec
        self.call_timeout = call_timeout
        self.oauth = oauth
        self.status = "idle"  # idle | connecting | ready | auth_required | error
        self.error = ""
        self.challenge: dict[str, str] = {}  # what the server's 401 said (resource_metadata, scope)
        self.fallback = False  # True when the old HTTP+SSE transport had to be used
        self.tools: list[dict[str, Any]] = []
        self.info: dict[str, Any] = {}
        self.capabilities: dict[str, Any] = {}
        self.transport: StdioTransport | HttpTransport | SseTransport | None = None
        self._lock = threading.Lock()

    def provider(self) -> Provider | None:
        if self.oauth is None or self.spec["transport"] == "stdio":
            return None
        return Provider(self.oauth, self.name, self.spec["url"])

    def _handshake(self, transport: StdioTransport | HttpTransport | SseTransport) -> None:
        transport.start()
        result = transport.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL,
                "capabilities": {"extensions": {UI_EXTENSION: {"mimeTypes": [UI_MIME]}}},
                "clientInfo": {"name": "fle-studio", "version": __version__},
            },
            CONNECT_TIMEOUT,
        )
        result = result if isinstance(result, dict) else {}
        self.info = result.get("serverInfo") if isinstance(result.get("serverInfo"), dict) else {}
        self.capabilities = result.get("capabilities") if isinstance(result.get("capabilities"), dict) else {}
        if isinstance(transport, (HttpTransport, SseTransport)):
            transport.version = str(result.get("protocolVersion") or PROTOCOL)
        transport.notify("notifications/initialized")
        self.tools = self._list_tools(transport)

    def connect(self) -> None:
        """Start the server and read its tool list; sets status and error instead of raising."""
        with self._lock:
            self.status, self.error, self.challenge, self.fallback = "connecting", "", {}, False
            kind = self.spec["transport"]
            provider = self.provider()
            transport: StdioTransport | HttpTransport | SseTransport
            if kind == "stdio":
                transport = StdioTransport(self.spec)
            elif kind == "sse":
                transport, self.fallback = SseTransport(self.spec, provider), True
            else:
                transport = HttpTransport(self.spec, provider)
            self.transport = transport
            try:
                try:
                    self._handshake(transport)
                except McpHttpError as exc:
                    # Not a 401/403: a server of the older revision answers a POST to its URL with 4xx (404/405/400).
                    if kind != "http" or exc.status in (401, 403) or not 400 <= exc.status < 500:
                        raise
                    transport.close()
                    transport = self.transport = SseTransport(self.spec, provider)
                    self.fallback = True
                    self._handshake(transport)
                self.status = "ready"
            except McpHttpError as exc:
                transport.close()
                self.tools = []
                if exc.status == 401 and self.oauth is not None:
                    self.status = "auth_required"
                    self.challenge = bearer_challenge(exc.www_authenticate)
                    self.error = "Sign in to use this server."
                else:
                    self.status, self.error = "error", str(exc)
            except (McpError, OSError, ValueError) as exc:
                self.status, self.error, self.tools = "error", str(exc), []
                transport.close()

    def _list_tools(self, transport: StdioTransport | HttpTransport | SseTransport) -> list[dict[str, Any]]:
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

    def _ready_transport(self) -> StdioTransport | HttpTransport | SseTransport:
        transport = self.transport
        if self.status != "ready" or transport is None:
            raise McpError(f"The server '{self.name}' is not connected" + (f": {self.error}" if self.error else ""))
        if not transport.alive():  # it exited since the last call: one new start, before anything was sent
            self.connect()
            transport = self.transport
            if self.status != "ready" or transport is None:
                raise McpError(f"The server '{self.name}' stopped and did not start again: {self.error}")
        return transport

    def _request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        transport = self._ready_transport()
        try:
            return transport.request(method, params, timeout)
        except McpHttpError as exc:
            if exc.status == 401 and self.oauth is not None:  # the sign-in ended while we were connected
                self.status, self.error = "auth_required", "Sign in again to use this server."
                self.challenge = bearer_challenge(exc.www_authenticate)
                raise McpError(self.error) from None
            raise

    def call_raw(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; the server's whole result (content, structuredContent, isError, _meta)."""
        result = self._request("tools/call", {"name": tool, "arguments": arguments}, self.call_timeout)
        return result if isinstance(result, dict) else {"content": [{"type": "text", "text": str(result or "")}]}

    def call(self, tool: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; (text, is_error). Raises McpError when the server cannot be reached."""
        return result_text(self.call_raw(tool, arguments))

    def read_resource(self, uri: str) -> dict[str, Any]:
        """A ``ui://`` resource for MCP Apps: {"html", "mime", "meta"}. Anything else is refused."""
        if not isinstance(uri, str) or not uri.startswith("ui://"):
            raise McpError("Only ui:// resources can be shown as apps")
        result = self._request("resources/read", {"uri": uri}, 30.0)
        contents = result.get("contents") if isinstance(result, dict) else None
        item = next((c for c in contents or [] if isinstance(c, dict) and c.get("uri", uri) == uri), None)
        if item is None:
            raise McpError(f"The server returned nothing for {uri}")
        mime = str(item.get("mimeType") or "")
        if mime.replace(" ", "").lower() != UI_MIME:
            raise McpError(f"{uri} is {mime or 'untyped'}, not {UI_MIME}")
        if isinstance(item.get("text"), str):
            html = item["text"]
        elif isinstance(item.get("blob"), str):
            try:
                html = base64.b64decode(item["blob"], validate=False).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                raise McpError(f"{uri} is not valid text") from None
        else:
            raise McpError(f"{uri} has no content")
        if len(html.encode("utf-8")) > MAX_APP_BYTES:
            raise McpError(f"{uri} is larger than {MAX_APP_BYTES // 1_000_000} MB")
        meta = item.get("_meta") if isinstance(item.get("_meta"), dict) else {}
        return {"html": html, "mime": UI_MIME, "meta": meta}

    def close(self) -> None:
        if self.transport is not None:
            with contextlib.suppress(Exception):
                self.transport.close()
        self.status = "idle"


# ------------------------------------------------------------------------------------------------ all servers


class McpManager:
    """The servers named in the settings. They connect in the background and stay connected until the settings
    change or FLE Studio ends."""

    def __init__(self, oauth: OAuthManager | None = None) -> None:
        self._servers: dict[str, McpServer] = {}
        self._lock = threading.RLock()
        self.oauth = oauth
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
                    server = self._servers[name] = McpServer(name, spec, call_timeout, self.oauth)
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
