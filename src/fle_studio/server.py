"""FLE Studio's local web server: the page, a JSON API, and an OpenAI-compatible proxy to the engine.

Safety model:
- By default it listens on 127.0.0.1 only. Requests must name a loopback host (or, in LAN mode, an IP address or this
  machine's name), which stops DNS-rebinding pages from talking to it.
- Requests that change something must come from the page itself: a cross-origin Origin header is refused, and /api
  calls carry a per-run token that only the served page knows.
- In LAN mode, other machines need the access key (a cookie after logging in, or a Bearer header for API clients).
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import ipaddress
import json
import mimetypes
import re
import secrets
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import __version__, bench, catalog, hf, launch, net, paths, service, system
from .chats import Chats
from .config import Config
from .downloads import Downloads
from .engine import Engine, EngineBusy
from .engines import Engines, variants
from .library import Library
from .metrics import Live

LOOPBACK_NAMES = {"127.0.0.1", "::1", "localhost"}
MAX_JSON = 4 * 1024 * 1024
MAX_PROXY = 64 * 1024 * 1024
STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon", ".webp": "image/webp",
                ".woff2": "font/woff2", ".json": "application/json", ".txt": "text/plain", ".md": "text/markdown"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _is_loopback(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        return False


class Studio:
    """Everything the server works with, created once per run."""

    def __init__(self, home: Path | None = None) -> None:
        self.data_dir = paths.ensure(home or paths.data_dir())
        self.config = Config(self.data_dir / "settings.json")
        self.downloads = Downloads(self.data_dir, token=lambda: self.config.get("hf_token"))
        self.engines = Engines(self.data_dir, self.downloads, self.config)
        self.library = Library(lambda: self.config.models_dir, self.data_dir)
        self.profiles = launch.Profiles(self.data_dir)
        self.engine = Engine(self.data_dir)
        self.engine.restart_on_crash = bool(self.config.get("launch.restart_on_crash"))
        self.live = Live(self.engine)
        self.chats = Chats(self.data_dir)
        self.bench = bench.Bench(self.engine, self.data_dir)
        self.token = secrets.token_urlsafe(24)
        self.lan = False
        self._sys: tuple[float, dict[str, Any] | None] = (0.0, None)
        self._sys_lock = threading.Lock()

    # ------------------------------------------------------------------------------------------ helpers

    def sysinfo(self, fresh: bool = False) -> dict[str, Any]:
        with self._sys_lock:
            at, cached = self._sys
            if cached is None or fresh or time.time() - at > 5:
                cached = system.info(self.config.models_dir)
                self._sys = (time.time(), cached)
            return cached

    def session_cookie(self) -> str:
        key = self.config.get("access_key")
        return hmac.new(key.encode(), b"fle-studio-session", hashlib.sha256).hexdigest() if key else ""

    def engine_for(self, want: str, info: dict[str, Any]) -> dict[str, Any]:
        eng = self.config.get("engine")
        custom = None
        if eng.get("custom_path"):
            try:
                custom = self.engines.resolve("custom", info)
            except RuntimeError:
                custom = None
        if want == "custom":
            if not custom:
                raise ApiError(400, "Set a working llama-server path on the Engine page")
            return custom
        if want in ("fle", "upstream") and custom and custom["flavor"] == want and eng.get("choice") == "custom":
            return custom
        try:
            return self.engines.resolve(want, info)
        except RuntimeError as exc:
            if custom and (want == "auto" or custom["flavor"] == want):
                return custom
            raise ApiError(409, str(exc)) from None

    def make_plan(self, model_ident: str, profile_ident: str | None) -> launch.Plan:
        model = self.library.get(model_ident) if model_ident else None
        settings = json.loads(json.dumps(self.config.data))
        prof_id = profile_ident or settings["launch"].get("profile") or "auto"
        info = self.sysinfo(fresh=True)
        port = launch.free_port(int(settings["launch"].get("port") or 0))
        try:
            if prof_id == "auto":
                if not model:
                    raise ApiError(404, "Choose a model first")
                if not model["complete"]:
                    raise ApiError(409, "This model is not completely downloaded yet")
                eng = self.engine_for("auto", info)
                return launch.auto_plan(model=model, engine=eng, settings=settings, sysinfo=info, port=port,
                                        data_dir=self.data_dir)
            prof = self.profiles.get(prof_id)
            if not prof:
                raise ApiError(404, f"No launch profile {prof_id}")
            if prof.engine == "custom":
                # Only to learn which flags its binary knows; the profile still starts that binary (cached per version).
                eng = self.engines.detect(prof.binary) if prof.binary and "{" not in prof.binary else None
            else:
                eng = self.engine_for(prof.engine, info)
            return launch.profile_plan(profile=prof, model=model, engine=eng, settings=settings, sysinfo=info,
                                       port=port, data_dir=self.data_dir)
        except launch.PlanError as exc:
            raise ApiError(400, str(exc)) from None

    def catalog_view(self) -> list[dict[str, Any]]:
        info = self.sysinfo()
        local = {m["catalog_id"]: m for m in self.library.scan() if m.get("catalog_id")}
        jobs = {j.get("dedupe"): j for j in self.downloads.list() if j["state"] not in ("done", "cancelled")}
        out = []
        for entry in catalog.catalog(info):
            entry["local"] = local.get(entry["id"])
            entry["job"] = jobs.get(f"catalog:{entry['id']}")
            out.append(entry)
        return out

    def extras_view(self) -> dict[str, Any]:
        root = self.config.models_dir
        out = {}
        for key, ref in (("mtp_head", catalog.MTP_HEAD), ("mtp_vocab", catalog.MTP_VOCAB)):
            dest = root / ref.repo / ref.path
            out[key] = {"name": ref.name, "size": ref.size, "present": dest.exists()}
        return out

    def download_catalog(self, ident: str, with_mtp: bool) -> dict[str, Any]:
        model = catalog.BY_ID.get(ident)
        if not model:
            raise ApiError(404, "Unknown catalog model")
        root = self.config.models_dir
        refs = list(model.files)
        if with_mtp and model.mtp:
            refs += [catalog.MTP_HEAD, catalog.MTP_VOCAB]
        files = [{"url": hf.resolve_url(r.repo, r.path), "dest": root / r.repo / r.path, "size": r.size,
                  "sha256": r.sha256} for r in refs]
        return self.downloads.add(title=model.name, files=files, kind="model", dedupe=f"catalog:{ident}",
                                  meta={"catalog_id": ident})

    def download_hf(self, repo: str, revision: str, wanted: list[str]) -> dict[str, Any]:
        token = self.config.get("hf_token")
        tree = {f["path"]: f for f in hf.tree(repo, revision, token)}
        missing = [p for p in wanted if p not in tree]
        if not wanted or missing:
            raise ApiError(400, "Pick files that exist in the repo" + (f": {missing[0]} does not" if missing else ""))
        root = self.config.models_dir
        files = []
        for p in wanted:
            f = tree[p]
            lfs = f.get("lfs") or {}
            files.append({"url": hf.resolve_url(repo, p, revision), "dest": root / repo / p,
                          "size": int(f.get("size") or 0), "sha256": lfs.get("oid")})
        title = f"{repo.split('/')[-1]} · {hf.quant_label(wanted[0].rsplit('/', 1)[-1])}"
        return self.downloads.add(title=title, files=files, kind="model",
                                  dedupe=f"hf:{repo}:{','.join(sorted(wanted))}")

    def state(self) -> dict[str, Any]:
        info = self.sysinfo()
        return {
            "app": "fle-studio",
            "version": __version__,
            "platform": info["platform"],
            "systemd": service.available(),
            "data_dir": str(self.data_dir),
            "lan": self.lan,
            "settings": self.config.public(),
            "system": info,
            "engine": self.engine.status(),
            "models": self.library.scan(),
            "catalog": self.catalog_view(),
            "extras": self.extras_view(),
            "profiles": [p.public() for p in self.profiles.all()],
            "engines": {"installed": self.engines.installed(), "variants": variants(info)},
            "downloads": self.downloads.list(),
            "reference": {"machine": catalog.REFERENCE_MACHINE, "report": catalog.REPORT_URL},
        }


# ---------------------------------------------------------------------------------------------- HTTP


ROUTES: list[tuple[str, re.Pattern[str], str]] = [(m, re.compile(p), n) for m, p, n in [
    ("GET", r"^/api/ping$", "ping"),
    ("GET", r"^/api/state$", "state"),
    ("GET", r"^/api/system$", "system"),
    ("GET", r"^/api/models$", "models"),
    ("POST", r"^/api/models/add$", "models_add"),
    ("POST", r"^/api/models/(?P<id>[0-9a-f]{12})/remove$", "models_remove"),
    ("GET", r"^/api/catalog$", "catalog"),
    ("POST", r"^/api/catalog/(?P<id>[\w.-]+)/download$", "catalog_download"),
    ("GET", r"^/api/hf/search$", "hf_search"),
    ("GET", r"^/api/hf/repo$", "hf_repo"),
    ("POST", r"^/api/hf/download$", "hf_download"),
    ("GET", r"^/api/downloads$", "downloads"),
    ("POST", r"^/api/downloads/(?P<id>[0-9a-f]+)/(?P<action>pause|resume|cancel|remove)$", "download_action"),
    ("GET", r"^/api/engines$", "engines"),
    ("POST", r"^/api/engines/install$", "engines_install"),
    ("POST", r"^/api/engines/check$", "engines_check"),
    ("POST", r"^/api/engines/(?P<id>[\w.-]+)/remove$", "engines_remove"),
    ("GET", r"^/api/engine$", "engine_status"),
    ("GET", r"^/api/engine/log$", "engine_log"),
    ("POST", r"^/api/engine/preview$", "engine_preview"),
    ("POST", r"^/api/engine/start$", "engine_start"),
    ("POST", r"^/api/engine/stop$", "engine_stop"),
    ("POST", r"^/api/engine/attach$", "engine_attach"),
    ("POST", r"^/api/engine/service/(?P<action>start|stop|restart)$", "engine_service"),
    ("GET", r"^/api/live$", "live"),
    ("GET", r"^/api/settings$", "settings"),
    ("POST", r"^/api/settings$", "settings_update"),
    ("GET", r"^/api/profiles$", "profiles"),
    ("POST", r"^/api/profiles$", "profiles_save"),
    ("POST", r"^/api/profiles/(?P<id>[\w.-]+)/delete$", "profiles_delete"),
    ("GET", r"^/api/chats$", "chats"),
    ("POST", r"^/api/chats$", "chat_create"),
    ("GET", r"^/api/chats/(?P<id>[0-9a-f]{16})$", "chat_get"),
    ("PUT", r"^/api/chats/(?P<id>[0-9a-f]{16})$", "chat_put"),
    ("DELETE", r"^/api/chats/(?P<id>[0-9a-f]{16})$", "chat_delete"),
    ("GET", r"^/api/bench$", "bench"),
    ("POST", r"^/api/bench/start$", "bench_start"),
    ("POST", r"^/api/bench/cancel$", "bench_cancel"),
    ("POST", r"^/api/bench/(?P<id>[0-9a-f]+)/delete$", "bench_delete"),
]]

LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FLE Studio · sign in</title>
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0b0e14;color:#e9edf3;
font:15px/1.5 system-ui,sans-serif}form{background:#10141c;border:1px solid #1f2633;border-radius:14px;padding:28px;
width:min(360px,90vw);display:grid;gap:12px}input{background:#151a24;border:1px solid #1f2633;border-radius:8px;
color:inherit;padding:10px 12px;font:inherit}button{background:#7aa7ff;color:#0b0e14;border:0;border-radius:8px;
padding:10px;font-weight:600;cursor:pointer}p{margin:0;color:#94a0b2;font-size:13px}.err{color:#ff8a8a}</style></head>
<body><form method="post" action="/login"><strong>FLE Studio</strong><p>This FLE Studio is open to the local network.
Enter its access key (Settings → Network on the machine that runs it).</p>__ERR__
<input type="password" name="key" autocomplete="current-password" autofocus placeholder="Access key">
<button type="submit">Sign in</button></form></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = f"FLEStudio/{__version__}"
    protocol_version = "HTTP/1.1"
    studio: Studio  # set on the subclass made by make_server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature from the base class
        return  # quiet; the engine log is what users read

    # ------------------------------------------------------------------------------------------ plumbing

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self, limit: int) -> bytes:
        """The request body, read once. The page sends `{}` with every POST, also where the handler wants nothing."""
        length = self._unread
        if length > limit:
            raise ApiError(413, "Request too large")
        self._unread = 0
        return self.rfile.read(length) if length else b""

    def _drain(self) -> None:
        """Read what the handler left of the body, or the next request on this connection would start with it."""
        length, self._unread = self._unread, 0
        if length > MAX_JSON:  # nobody wants a large upload here: end the connection instead of reading it
            self.close_connection = True
        elif length:
            try:
                self.rfile.read(length)
            except OSError:
                self.close_connection = True

    def read_json(self) -> dict[str, Any]:
        raw = self._body(MAX_JSON)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ApiError(400, "Invalid JSON") from None
        if not isinstance(data, dict):
            raise ApiError(400, "Expected a JSON object")
        return data

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        name = host[1:].split("]", 1)[0] if host.startswith("[") else host.rsplit(":", 1)[0]
        name = name.lower()
        if name in LOOPBACK_NAMES:
            return True
        if not self.studio.lan:
            return False
        if _is_ip(name):
            return True
        own = socket.gethostname().lower()
        allowed = {own, own.split(".")[0], *(h.strip().lower() for h in
                                            str(self.studio.config.get("server").get("allowed_hosts", "")).split(",")
                                            if h.strip())}
        return name in allowed or any(name.startswith(a + ".") for a in allowed if a)

    def _authed(self) -> bool:
        if _is_loopback(self.client_address[0]):
            return True
        expected = self.studio.session_cookie()
        if not expected:
            return False
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), self.studio.config.get("access_key")):
            return True
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "fle_session" and hmac.compare_digest(v, expected):
                return True
        return False

    def _guard(self, path: str) -> bool:
        if not self._host_ok():
            self._send(403, b"FLE Studio: this host name is not allowed.", "text/plain; charset=utf-8")
            return False
        if path == "/login":
            return True
        if not self._authed():
            if path.startswith(("/api/", "/v1/")):
                self.send_json(401, {"error": {"message": "Sign in with the access key", "type": "auth"}})
            else:
                self._send(303, b"", "text/plain", {"Location": "/login"})
            return False
        if self.command in ("POST", "PUT", "DELETE", "PATCH"):
            origin = self.headers.get("Origin")
            if origin and origin != "null" and urlparse(origin).netloc != self.headers.get("Host"):
                self.send_json(403, {"error": {"message": "Cross-origin request refused"}})
                return False
            if origin == "null":
                self.send_json(403, {"error": {"message": "Request from an opaque origin refused"}})
                return False
            if path.startswith("/api/") and not hmac.compare_digest(self.headers.get("X-FLE-Token", ""),
                                                                    self.studio.token):
                # Usually a page from before a restart of Studio; the header tells it to reload for the new token.
                self._send(403, json.dumps({"error": {"message": "Session expired; reload the page"}}).encode(),
                           "application/json; charset=utf-8", {"X-FLE-Session": "expired"})
                return False
        return True

    def _dispatch(self) -> None:
        url = urlparse(self.path)
        path = url.path
        try:
            self._unread = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            self._unread, self.close_connection = 0, True
        try:
            if not self._guard(path):
                return
            if path.startswith("/v1/"):
                self.proxy()
                return
            if path == "/login":
                self.login()
                return
            for method, rx, name in ROUTES:
                m = rx.match(path)
                if m and method == self.command:
                    query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                    getattr(self, "api_" + name)(query=query, **m.groupdict())
                    return
            if self.command in ("GET", "HEAD") and not path.startswith("/api/"):
                self.static(path)
                return
            self.send_json(404, {"error": {"message": "Not found"}})
        except ApiError as exc:
            self.send_json(exc.status, {"error": {"message": exc.message}})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except EngineBusy as exc:
            self.send_json(409, {"error": {"message": str(exc)}})
        except (KeyError, FileNotFoundError):
            self.send_json(404, {"error": {"message": "Not found"}})
        except (ValueError, RuntimeError, OSError) as exc:
            self.send_json(400, {"error": {"message": str(exc)}})
        except Exception as exc:  # last resort: report instead of dropping the connection
            traceback.print_exc()
            self.send_json(500, {"error": {"message": f"Internal error: {exc}"}})
        finally:
            self._drain()

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _dispatch

    # ------------------------------------------------------------------------------------------ pages

    def static(self, path: str) -> None:
        name = "index.html" if path in ("/", "/index.html") else path.lstrip("/")
        if not re.match(r"^[\w][\w./-]*$", name) or ".." in name:
            raise ApiError(404, "Not found")
        node = resources.files("fle_studio").joinpath("web", *name.split("/"))
        try:
            data = node.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError):
            if "." not in name.rsplit("/", 1)[-1]:  # client-side routes fall back to the app
                node = resources.files("fle_studio").joinpath("web", "index.html")
                data, name = node.read_bytes(), "index.html"
            else:
                raise ApiError(404, "Not found") from None
        if name == "index.html":
            data = data.replace(b"__FLE_TOKEN__", self.studio.token.encode())
        # Our own table first: on Windows `mimetypes` reads the registry, where .js is often text/plain, and
        # browsers refuse to run module scripts served like that.
        ctype = (STATIC_TYPES.get(Path(name).suffix.lower()) or mimetypes.guess_type(name)[0]
                 or "application/octet-stream")
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        csp = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
               "script-src 'self'; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self._send(200, data, ctype, {"Content-Security-Policy": csp, "X-Frame-Options": "DENY",
                                      "X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"})

    def login(self) -> None:
        if self.command == "POST":
            form = parse_qs(self._body(4096).decode("utf-8", "replace"))
            key = (form.get("key") or [""])[-1]
            real = self.studio.config.get("access_key")
            if real and hmac.compare_digest(key, real):
                cookie = (f"fle_session={self.studio.session_cookie()}; Path=/; HttpOnly; SameSite=Strict; "
                          "Max-Age=2592000")
                self._send(303, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})
                return
            time.sleep(1.0)  # slow down guessing
            page = LOGIN_PAGE.replace("__ERR__", '<p class="err">That key is not right.</p>')
            self._send(401, page.encode(), "text/html; charset=utf-8")
            return
        self._send(200, LOGIN_PAGE.replace("__ERR__", "").encode(), "text/html; charset=utf-8")

    # ------------------------------------------------------------------------------------------ proxy

    def proxy(self) -> None:
        """Pass /v1/* to the engine; streams (SSE) are forwarded as they arrive."""
        state, base, headers = self.studio.engine.target()[:3]
        if state not in ("ready", "attached") or not base:
            self.send_json(503, {"error": {"message": "No model is loaded. Load one in FLE Studio first.",
                                           "type": "unavailable_error", "code": 503}})
            return
        body = self._body(MAX_PROXY) or None
        u = urlparse(base)
        if u.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                u.hostname or "", u.port or 443, timeout=900, context=net.ssl_context())
        else:
            conn = http.client.HTTPConnection(u.hostname or "", u.port or 80, timeout=900)
        prefix = u.path.rstrip("/")
        fwd = {"Content-Type": self.headers.get("Content-Type", "application/json"),
               "Accept": self.headers.get("Accept", "*/*"), **headers}
        try:
            conn.request(self.command, prefix + self.path, body=body, headers=fwd)
            resp = conn.getresponse()
        except OSError as exc:
            conn.close()
            self.send_json(502, {"error": {"message": f"The engine did not answer: {exc}", "type": "bad_gateway"}})
            return
        ctype = resp.getheader("Content-Type", "application/json")
        if "text/event-stream" not in ctype:
            data = resp.read()
            conn.close()
            self._record(data)
            self._send(resp.status, data, ctype)
            return
        self.send_response(resp.status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        tail = b""
        try:
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                tail = (tail + chunk)[-262144:]
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # the browser stopped reading (Stop button); closing our side cancels the generation
        finally:
            conn.close()
        self._record_stream(tail)

    def _record(self, data: bytes) -> None:
        try:
            obj = json.loads(data.decode("utf-8"))
        except ValueError:
            return
        if isinstance(obj, dict) and obj.get("timings"):
            self.studio.live.record_response(obj["timings"], obj.get("usage"))

    def _record_stream(self, tail: bytes) -> None:
        for line in reversed(tail.decode("utf-8", "replace").splitlines()):
            if line.startswith("data:") and '"timings"' in line:
                try:
                    obj = json.loads(line[5:].strip())
                except ValueError:
                    continue
                self.studio.live.record_response(obj.get("timings") or {}, obj.get("usage"))
                return

    # ------------------------------------------------------------------------------------------ API

    def api_ping(self, query: dict[str, str]) -> None:
        self.send_json(200, {"app": "fle-studio", "version": __version__})

    def api_state(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.state())

    def api_system(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.sysinfo(fresh=True))

    def api_models(self, query: dict[str, str]) -> None:
        self.send_json(200, {"models": self.studio.library.scan(), "extras": self.studio.extras_view()})

    def api_models_add(self, query: dict[str, str]) -> None:
        body = self.read_json()
        try:
            self.send_json(200, self.studio.library.add(str(body.get("path") or "")))
        except Exception as exc:  # GGUFError and friends: tell the user why
            raise ApiError(400, str(exc)) from None

    def api_models_remove(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        body = self.read_json()
        status = self.studio.engine.status()
        if body.get("delete") and status["model"].get("id") == id and status["state"] in ("starting", "ready"):
            raise ApiError(409, "Eject the model before deleting its files")
        self.studio.library.remove(id, delete_files=bool(body.get("delete")))
        self.send_json(200, {"ok": True})

    def api_catalog(self, query: dict[str, str]) -> None:
        self.send_json(200, {"catalog": self.studio.catalog_view(), "extras": self.studio.extras_view()})

    def api_catalog_download(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        body = self.read_json()
        self.send_json(200, self.studio.download_catalog(id, bool(body.get("mtp", True))))

    def api_hf_search(self, query: dict[str, str]) -> None:
        q = query.get("q", "").strip()
        if len(q) < 2:
            self.send_json(200, {"results": []})
            return
        self.send_json(200, {"results": hf.search(q, token=self.studio.config.get("hf_token"))})

    def api_hf_repo(self, query: dict[str, str]) -> None:
        repo, revision, hint = hf.parse_repo(query.get("repo", ""))
        token = self.studio.config.get("hf_token")
        try:
            files = hf.tree(repo, revision, token)
            info = hf.repo_info(repo, token)
        except Exception as exc:  # HTTP 401/404 and network errors become one readable message
            raise ApiError(404, f"Could not open {repo}: {getattr(exc, 'reason', exc)}") from None
        groups = hf.gguf_groups(files, hint)
        sysinfo = self.studio.sysinfo()
        for g in groups:
            g["fit"] = catalog.fit(g["size"], sysinfo) if g["kind"] == "model" else None
        self.send_json(200, {"repo": repo, "revision": revision, "info": info, "groups": groups})

    def api_hf_download(self, query: dict[str, str]) -> None:
        body = self.read_json()
        repo, revision, _ = hf.parse_repo(str(body.get("repo") or ""))
        files = [str(p) for p in body.get("files") or []]
        self.send_json(200, self.studio.download_hf(repo, str(body.get("revision") or revision), files))

    def api_downloads(self, query: dict[str, str]) -> None:
        self.send_json(200, {"downloads": self.studio.downloads.list()})

    def api_download_action(self, query: dict[str, str], id: str, action: str) -> None:  # noqa: A002
        self.send_json(200, self.studio.downloads.action(id, action))

    def api_engines(self, query: dict[str, str]) -> None:
        info = self.studio.sysinfo()
        self.send_json(200, {"installed": self.studio.engines.installed(), "variants": variants(info)})

    def api_engines_install(self, query: dict[str, str]) -> None:
        body = self.read_json()
        variant = str(body.get("variant") or "")
        if variant not in {v["id"] for v in variants(self.studio.sysinfo())}:
            raise ApiError(400, "That engine does not fit this machine")
        try:
            self.send_json(200, self.studio.engines.install(variant, self.studio.sysinfo(fresh=True)))
        except RuntimeError as exc:
            raise ApiError(409, str(exc)) from None
        except OSError as exc:
            raise ApiError(502, f"Could not reach GitHub: {getattr(exc, 'reason', exc)}") from None

    def api_engines_check(self, query: dict[str, str]) -> None:
        body = self.read_json()
        self.send_json(200, self.studio.engines.detect(str(body.get("path") or "")))

    def api_engines_remove(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.studio.engines.remove(id)
        self.send_json(200, {"ok": True})

    def api_engine_status(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.engine.status())

    def api_engine_log(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.engine.log_since(int(query.get("since") or 0)))

    def api_engine_preview(self, query: dict[str, str]) -> None:
        body = self.read_json()
        plan = self.studio.make_plan(str(body.get("model") or ""), body.get("profile"))
        self.send_json(200, plan.public())

    def api_engine_start(self, query: dict[str, str]) -> None:
        body = self.read_json()
        model_ident = str(body.get("model") or self.studio.config.get("last_model") or "")
        profile = body.get("profile") or self.studio.config.get("launch.profile")
        plan = self.studio.make_plan(model_ident, profile)
        self.studio.engine.restart_on_crash = bool(self.studio.config.get("launch.restart_on_crash"))
        self.studio.engine.start(plan)
        self.studio.config.update({"last_model": model_ident, "launch": {"profile": profile or "auto"}})
        self.send_json(200, self.studio.engine.status())

    def api_engine_stop(self, query: dict[str, str]) -> None:
        if self.studio.engine.bound():  # Disconnect: do not connect again at the next start
            self.studio.config.update({"attach_auto": False})
        self.studio.engine.stop()
        self.send_json(200, self.studio.engine.status())

    def api_engine_attach(self, query: dict[str, str]) -> None:
        body = self.read_json()
        url = str(body.get("url") or "")
        unit = str(body.get("service") or "").strip()
        key = body.get("key")
        patch: dict[str, Any] = {"attach_url": url, "attach_service": unit}
        if key is not None:
            patch["attach_key"] = str(key)
        self.studio.config.update(patch)
        status = self.studio.engine.attach(url, self.studio.config.get("attach_key"), unit)
        self.studio.config.update({"attach_auto": True})
        self.send_json(200, status)

    def api_engine_service(self, query: dict[str, str], action: str) -> None:
        self.send_json(200, self.studio.engine.service_action(action))

    def api_live(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.live.snapshot(float(query.get("since") or 0)))

    def api_settings(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.config.public())

    def api_settings_update(self, query: dict[str, str]) -> None:
        body = self.read_json()
        result = self.studio.config.update(body)
        self.studio.engine.restart_on_crash = bool(self.studio.config.get("launch.restart_on_crash"))
        self.send_json(200, result)

    def api_profiles(self, query: dict[str, str]) -> None:
        self.send_json(200, {"profiles": [p.public() for p in self.studio.profiles.all()]})

    def api_profiles_save(self, query: dict[str, str]) -> None:
        try:
            prof = self.studio.profiles.save(self.read_json())
        except launch.PlanError as exc:
            raise ApiError(400, str(exc)) from None
        self.send_json(200, prof.public())

    def api_profiles_delete(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.studio.profiles.delete(id)
        if self.studio.config.get("launch.profile") == id:
            self.studio.config.update({"launch": {"profile": "auto"}})
        self.send_json(200, {"ok": True})

    def api_chats(self, query: dict[str, str]) -> None:
        self.send_json(200, {"chats": self.studio.chats.list()})

    def api_chat_create(self, query: dict[str, str]) -> None:
        self.send_json(200, self.studio.chats.save(None, self.read_json()))

    def api_chat_get(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.send_json(200, self.studio.chats.get(id))

    def api_chat_put(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.send_json(200, self.studio.chats.save(id, self.read_json()))

    def api_chat_delete(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.studio.chats.delete(id)
        self.send_json(200, {"ok": True})

    def api_bench(self, query: dict[str, str]) -> None:
        self.send_json(200, {"current": self.studio.bench.status(), "results": self.studio.bench.results(),
                             "tests": [t["name"] for t in bench.TESTS]})

    def api_bench_start(self, query: dict[str, str]) -> None:
        body = self.read_json()
        st = self.studio.engine.status()
        info = self.studio.sysinfo()
        gpu = ", ".join(f"{g['name']} {g['memory_total'] / 2**30:.0f} GB" for g in info.get("gpus", []))
        svc = st.get("service") or {}
        context = {"model": st["model"].get("name"), "profile": (st.get("plan") or {}).get("profile") or
                   (f"service {svc['unit']}" if svc else "attached" if st["attached"] else ""),
                   "engine": (st.get("plan") or {}).get("engine") or "",
                   "build": st.get("build"), "n_ctx": st.get("n_ctx"),
                   "machine": f"{info['cpu']['name']}, {info['memory']['total'] / 2**30:.0f} GB RAM" +
                              (f", {gpu}" if gpu else ""),
                   "command": (st.get("plan") or {}).get("command") or svc.get("command")}
        try:
            self.send_json(200, self.studio.bench.start(runs=int(body.get("runs") or 1),
                                                       label=str(body.get("label") or ""), context=context))
        except RuntimeError as exc:
            raise ApiError(409, str(exc)) from None

    def api_bench_cancel(self, query: dict[str, str]) -> None:
        self.studio.bench.cancel()
        self.send_json(200, {"ok": True})

    def api_bench_delete(self, query: dict[str, str], id: str) -> None:  # noqa: A002
        self.studio.bench.delete(id)
        self.send_json(200, {"ok": True})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False if paths.IS_WIN else True

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return  # a browser closed its connection; nothing to report
        super().handle_error(request, client_address)


def make_server(studio: Studio, host: str, port: int, tries: int = 20) -> Server:
    """Bind to host:port, or the next free port above it."""
    handler = type("BoundHandler", (Handler,), {"studio": studio})
    studio.lan = host not in ("127.0.0.1", "localhost", "::1")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    Server.address_family = family
    last: OSError | None = None
    for p in range(port, port + tries) if port else [0]:
        try:
            return Server((host, p), handler)
        except OSError as exc:
            last = exc
    raise OSError(f"No free port from {port} to {port + tries - 1}: {last}")
