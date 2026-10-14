"""Host and client mode: one FLE Studio can compute for another.

Host side (the machine with the GPU, `fle-studio --host 0.0.0.0` or a service):
  * TLS with a self-made certificate, see tlsgen.py. Its SHA-256 fingerprint is shown on the host and pinned by every
    client when it pairs (trust on first use). A changed fingerprint is refused and reported.
  * Pairing: the host shows a short-lived code (8 characters, 10 minutes, one use). A client proves that it knows the
    code with an HMAC bound to the fingerprint it saw, so a man in the middle neither learns the code nor gets a token.
    The reward is a client token of its own. The host stores only a hash of it, lists the paired clients and can
    revoke each one. Tokens are never logged.
  * Rights per client: the default may chat, look and measure; "admin" may also change settings, install engines and
    load or delete models. Installs and updates additionally ask for a confirmation on the client.
  * The access key of the browser route stays as it is.

Client side (the Mac with the browser):
  * Hosts are added by address and code, or by an SSH target; for the second Studio starts the system `ssh -N -L`,
    watches it and starts it again when it dies. No passwords: keys or the agent only.
  * The page talks only to its own local Studio. Everything under `/h/<host id>/` is passed on to the host, streams
    included, with the client token. Nothing here lists endpoints, so pages added later work on a host too.
  * Chats live on the host. A host can be mirrored for reading offline.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import json
import os
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import paths, tlsgen

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I
CODE_LENGTH = 8
CODE_TTL = 600.0
FAILS_PER_ADDRESS = 5
FAILS_TOTAL = 20
FAIL_WINDOW = 600.0
CODE_BURNED_AFTER = 10  # wrong tries (all addresses together) after which the code is thrown away
TOKEN_PREFIX = "fcl_"
DEFAULT_PORT = 8642
COLORS = ["#22d3ee", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#fb7185", "#60a5fa", "#a3e635"]

# Client tokens never reach these: a paired client must not manage clients or hosts, nor chain to other hosts.
OWNER_ONLY = re.compile(r"^/(?:api/(?:hosts|remote)(?:/|$)|h/)")
# What the default right allows beyond reading: chat, measure, and computing a command preview.
STANDARD_WRITES = re.compile(
    r"^/(?:api/chats(?:/|$)|api/attachments$|api/bench(?:/|$)|api/engine/preview$|api/flags/explain$|v1/)"
)
# Needs admin and, on top, a confirmation that the person at the client gave.
NEEDS_CONFIRM = re.compile(r"^/api/(?:engines/install|engines/[\w.-]+/remove|update/apply|models/[0-9a-f]{12}/remove)$")


class RemoteError(Exception):
    def __init__(self, status: int, message: str, code: str = "") -> None:
        super().__init__(message)
        self.status, self.message, self.code = status, message, code


class FingerprintError(OSError):
    def __init__(self, seen: str, expected: str) -> None:
        super().__init__("The host's certificate is not the one that was pinned")
        self.seen, self.expected = seen, expected


# ------------------------------------------------------------------------------------------------ small helpers


def _now() -> float:
    return time.time()


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write(path: Path, data: Any) -> None:
    paths.ensure(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    if not paths.IS_WIN:
        os.chmod(tmp, 0o600)  # holds tokens
    tmp.replace(path)


def normalize_code(code: str) -> str:
    return "".join(c for c in str(code).upper() if c in CODE_ALPHABET)


def pair_proof(code: str, fingerprint: str) -> str:
    """What a client sends instead of the code: bound to the certificate it saw."""
    msg = ("fle-pair|" + tlsgen.normalize(fingerprint)).encode()
    return hmac.new(normalize_code(code).encode(), msg, hashlib.sha256).hexdigest()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def host_name(studio: Any) -> str:
    return str(studio.config.get("remote.name") or "").strip() or socket.gethostname().split(".")[0]


def clean_name(text: Any, fallback: str) -> str:
    out = re.sub(r"[^\w .@()+'-]", "", str(text or "")).strip()[:60]
    return out or fallback


# ------------------------------------------------------------------------------------------------ host: clients


class Clients:
    """The paired clients and the pairing code, in `clients.json` and `pairing.json` of the data folder."""

    def __init__(self, data_dir: Path) -> None:
        self.file = data_dir / "clients.json"
        self.code_file = data_dir / "pairing.json"
        self._lock = threading.Lock()
        self._fails: dict[str, list[float]] = {}
        self._seen_saved = 0.0

    # ---- the code

    def new_code(self) -> dict[str, Any]:
        code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))
        expires = _now() + CODE_TTL
        with self._lock:
            _write(self.code_file, {"code": code, "expires": expires, "fails": 0})
        return {"code": code, "pretty": f"{code[:4]}-{code[4:]}", "expires": expires, "ttl": int(CODE_TTL)}

    def code_state(self) -> dict[str, Any]:
        data = _read(self.code_file, {})
        if data.get("code") and float(data.get("expires") or 0) > _now():
            return {"active": True, "expires": float(data["expires"])}
        return {"active": False, "expires": 0}

    def _fail(self, address: str) -> None:
        now = _now()
        for key in (address, "*"):
            self._fails[key] = [t for t in self._fails.get(key, []) if now - t < FAIL_WINDOW] + [now]

    def _limited(self, address: str) -> bool:
        now = _now()
        mine = [t for t in self._fails.get(address, []) if now - t < FAIL_WINDOW]
        every = [t for t in self._fails.get("*", []) if now - t < FAIL_WINDOW]
        return len(mine) >= FAILS_PER_ADDRESS or len(every) >= FAILS_TOTAL

    # ---- pairing

    def pair(self, proof: str, name: str, address: str, fingerprint: str) -> dict[str, Any]:
        with self._lock:
            if self._limited(address):
                raise RemoteError(429, "Too many wrong tries. Wait a few minutes.", "rate_limited")
            data = _read(self.code_file, {})
            code = str(data.get("code") or "")
            if not code or float(data.get("expires") or 0) <= _now():
                self._fail(address)
                raise RemoteError(
                    400, "There is no pairing code, or it has expired. Create a new one on the host.", "no_code"
                )
            expected = pair_proof(code, fingerprint)
            if not hmac.compare_digest(str(proof), expected):
                self._fail(address)
                data["fails"] = int(data.get("fails") or 0) + 1
                if data["fails"] >= CODE_BURNED_AFTER:
                    self.code_file.unlink(missing_ok=True)  # somebody is guessing: the code is gone
                else:
                    _write(self.code_file, data)
                raise RemoteError(
                    400, "The code is not right (or the host's certificate changed on the way).", "bad_code"
                )
            self.code_file.unlink(missing_ok=True)  # one use
            token = TOKEN_PREFIX + secrets.token_urlsafe(32)
            client = {
                "id": secrets.token_hex(6),
                "name": clean_name(name, "Client"),
                "hash": token_hash(token),
                "admin": False,
                "created": _now(),
                "last_seen": 0.0,
                "last_address": address,
            }
            clients = _read(self.file, [])
            clients.append(client)
            _write(self.file, clients)
        return {"token": token, "client": self.public(client)}

    # ---- the list

    @staticmethod
    def public(c: dict[str, Any]) -> dict[str, Any]:
        return {k: c[k] for k in ("id", "name", "admin", "created", "last_seen", "last_address") if k in c}

    def list(self) -> list[dict[str, Any]]:
        return [self.public(c) for c in _read(self.file, [])]

    def find(self, token: str, address: str = "") -> dict[str, Any] | None:
        want = token_hash(token)
        found = None
        for c in _read(self.file, []):
            if hmac.compare_digest(str(c.get("hash", "")), want):
                found = c
        if found is not None and _now() - float(found.get("last_seen") or 0) > 60:
            with self._lock:  # a record of "last seen" once a minute is enough
                clients = _read(self.file, [])
                for c in clients:
                    if c["id"] == found["id"]:
                        c["last_seen"], c["last_address"] = _now(), address
                _write(self.file, clients)
        return found

    def revoke(self, client_id: str) -> bool:
        with self._lock:
            clients = _read(self.file, [])
            kept = [c for c in clients if c["id"] != client_id]
            if len(kept) == len(clients):
                return False
            _write(self.file, kept)
        return True

    def set_admin(self, client_id: str, admin: bool) -> bool:
        with self._lock:
            clients = _read(self.file, [])
            hit = False
            for c in clients:
                if c["id"] == client_id:
                    c["admin"], hit = bool(admin), True
            if hit:
                _write(self.file, clients)
        return hit


def check_rights(client: dict[str, Any], method: str, path: str, confirmed: bool) -> tuple[int, str, str] | None:
    """None when the request is allowed, else (status, message, code)."""
    if path == "/api/remote/whoami":
        return None
    if OWNER_ONLY.match(path):
        return 403, "This can only be done on the host itself.", "owner_only"
    if method in ("GET", "HEAD"):
        return None
    if STANDARD_WRITES.match(path):
        return None
    if not client.get("admin"):
        return (
            403,
            (
                "This client may chat, look and measure. Changes need admin rights; "
                "the owner of the host can grant them."
            ),
            "no_admin",
        )
    if NEEDS_CONFIRM.match(path) and not confirmed:
        return 428, "Confirm this on the client.", "confirm"
    return None


# ------------------------------------------------------------------------------------------------ client: ssh tunnel


def ssh_program() -> list[str]:
    """The ssh command: `FLE_STUDIO_SSH` (a path, or a JSON list; for tests), else the system's ssh."""
    override = os.environ.get("FLE_STUDIO_SSH", "").strip()
    if override.startswith("["):
        try:
            parts = json.loads(override)
            if isinstance(parts, list) and parts:
                return [str(p) for p in parts]
        except ValueError:
            pass
    if override:
        return [override]
    found = shutil.which("ssh")
    if not found:
        raise RemoteError(400, "The ssh program was not found. Install the OpenSSH client.", "no_ssh")
    return [found]


_TARGET = re.compile(r"^(?:[A-Za-z0-9._-]+@)?[A-Za-z0-9._:-]+$")


def check_ssh(target: str, port: Any, remote_port: Any) -> tuple[str, int, int]:
    target = str(target or "").strip()
    if not target or target.startswith("-") or not _TARGET.match(target):
        raise RemoteError(400, "The SSH target must look like user@host (letters, digits, dots, dashes).")
    try:
        p = int(22 if port in (None, "") else port)
        rp = int(DEFAULT_PORT if remote_port in (None, "") else remote_port)
    except (TypeError, ValueError):
        raise RemoteError(400, "Ports must be numbers.") from None
    if not (1 <= p <= 65535 and 1 <= rp <= 65535):
        raise RemoteError(400, "Ports must be between 1 and 65535.")
    return target, p, rp


def ssh_command(program: list[str], target: str, port: int, local_port: int, remote_port: int) -> list[str]:
    """`ssh -N -L 127.0.0.1:<local>:127.0.0.1:<remote> …` with keys only, and a dead connection ends the process."""
    return [
        *program,
        "-N",
        "-T",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",  # never ask for a password: keys or the agent
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=10",
        target,
    ]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


class Tunnel:
    """`ssh -N -L` kept alive: started on first use, watched, started again with a growing pause when it dies."""

    def __init__(self, target: str, port: int, remote_port: int) -> None:
        self.target, self.port, self.remote_port = target, port, remote_port
        self.local_port = 0
        self.state = "stopped"  # stopped | starting | up | down
        self.error = ""
        self._proc: subprocess.Popen[bytes] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=8)
        self._restart = threading.Event()
        self._wake = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self.state = "starting"
            self._thread = threading.Thread(target=self._run, daemon=True, name="fle-ssh-tunnel")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self.state = "stopped"

    def restart(self) -> None:
        """A new ssh process now, without the pause (the 'Reconnect' button)."""
        self._restart.set()
        self._wake.set()
        proc = self._proc
        self.state = "starting"
        if proc is not None and proc.poll() is None:
            proc.terminate()  # the supervisor starts a new one
        self.start()

    def wait_ready(self, timeout: float = 15.0) -> bool:
        self.start()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.state == "up":
                return True
            if self.state == "down":
                return False  # the reason is in .error; the supervisor keeps trying in the background
            time.sleep(0.05)
        return self.state == "up"

    def info(self) -> dict[str, Any]:
        return {"state": self.state, "error": self.error, "local_port": self.local_port if self.state == "up" else 0}

    def _drain(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line:
                self._stderr.append(line)

    def _run(self) -> None:
        pause = 1.0
        while not self._stop.is_set():
            try:
                program = ssh_program()
            except RemoteError as exc:
                self.state, self.error = "down", exc.message
                return
            local = _free_port()
            self._stderr.clear()
            try:
                proc = subprocess.Popen(
                    ssh_command(program, self.target, self.port, local, self.remote_port),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                self.state, self.error = "down", f"Could not start ssh: {exc}"
                if self._stop.wait(pause):
                    return
                pause = min(pause * 2, 30)
                continue
            self._proc = proc
            threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
            self.state = "starting"
            deadline = time.monotonic() + 20
            while not self._stop.is_set() and proc.poll() is None and time.monotonic() < deadline:
                if _listening(local):
                    self.local_port, self.state, self.error, pause = local, "up", "", 1.0
                    break
                time.sleep(0.1)
            while not self._stop.is_set() and proc.poll() is None:
                time.sleep(0.3)
            if self._stop.is_set():
                if proc.poll() is None:
                    proc.terminate()
                return
            if proc.poll() is None:
                proc.terminate()  # never became ready
            if self._restart.is_set():  # asked for a new one: no pause, and not "down" in between
                self._restart.clear()
                self._wake.clear()
                continue
            time.sleep(0.1)  # let the reader see the last lines
            self.state = "down"
            self.error = " ".join(self._stderr) or f"ssh ended (code {proc.poll()})"
            self._wake.wait(pause)
            self._wake.clear()
            self._restart.clear()
            if self._stop.is_set():
                return
            pause = min(pause * 2, 30)


# ------------------------------------------------------------------------------------------------ client: to a host


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """TLS to a host whose certificate is pinned by fingerprint instead of checked against authorities."""

    def __init__(self, host: str, port: int, pinned: str, timeout: float) -> None:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        super().__init__(host, port, timeout=timeout, context=ctx)
        self.pinned = tlsgen.normalize(pinned)
        self.seen = ""

    def connect(self) -> None:
        super().connect()
        der = self.sock.getpeercert(binary_form=True) if isinstance(self.sock, ssl.SSLSocket) else b""
        self.seen = tlsgen.fingerprint_of(der) if der else ""
        if self.pinned and self.seen != self.pinned:
            self.close()
            raise FingerprintError(self.seen, self.pinned)


def parse_address(text: str) -> tuple[str, str, int]:
    """'gpu-box', 'gpu-box:8642', 'https://10.0.0.5:9000' or 'http://…' to (scheme, host, port)."""
    raw = str(text or "").strip()
    if not raw:
        raise RemoteError(400, "Enter the host's address.")
    if "://" not in raw:
        raw = "https://" + raw
    u = urlparse(raw)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise RemoteError(400, "The address must look like host or host:port.")
    try:
        port = u.port or DEFAULT_PORT
    except ValueError:
        raise RemoteError(400, "The port is not a number.") from None
    return u.scheme, u.hostname, port


def _resolve(name: str) -> str:
    """The IP address of a host name. The host accepts any IP in its Host header, but only some names."""
    try:
        return str(socket.getaddrinfo(name, None, type=socket.SOCK_STREAM)[0][4][0])
    except OSError as exc:
        raise OSError(f"Cannot find {name}: {exc}") from None


class Hosts:
    """The hosts this Studio can switch to, in `hosts.json` (with their client tokens, readable by the user only)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.file = data_dir / "hosts.json"
        self._lock = threading.Lock()
        self._tunnels: dict[str, Tunnel] = {}

    def all(self) -> list[dict[str, Any]]:
        return _read(self.file, [])

    def get(self, host_id: str) -> dict[str, Any]:
        for h in self.all():
            if h["id"] == host_id:
                return h
        raise RemoteError(404, "This host is not in the list.")

    def save(self, record: dict[str, Any]) -> None:
        with self._lock:
            hosts = [h for h in self.all() if h["id"] != record["id"]]
            hosts.append(record)
            hosts.sort(key=lambda h: h.get("added", 0))
            _write(self.file, hosts)

    def remove(self, host_id: str) -> None:
        tunnel = self._tunnels.pop(host_id, None)
        if tunnel:
            tunnel.stop()
        with self._lock:
            _write(self.file, [h for h in self.all() if h["id"] != host_id])
        shutil.rmtree(self.data_dir / "mirror" / host_id, ignore_errors=True)

    def public(self, h: dict[str, Any]) -> dict[str, Any]:
        out = {
            k: h.get(k)
            for k in ("id", "name", "color", "kind", "scheme", "address", "fingerprint", "mirror", "ssh", "paired")
        }
        out["fingerprint_pretty"] = tlsgen.pretty(h["fingerprint"]) if h.get("fingerprint") else ""
        tunnel = self._tunnels.get(h["id"])
        out["tunnel"] = tunnel.info() if tunnel else None
        return out

    def next_color(self) -> str:
        used = [h.get("color") for h in self.all()]
        for c in COLORS:
            if c not in used:
                return c
        return COLORS[len(used) % len(COLORS)]

    def tunnel(self, h: dict[str, Any]) -> Tunnel:
        t = self._tunnels.get(h["id"])
        if t is None:
            ssh = h["ssh"]
            t = self._tunnels[h["id"]] = Tunnel(ssh["target"], int(ssh["port"]), int(ssh["remote_port"]))
        return t

    def shutdown(self) -> None:
        for t in self._tunnels.values():
            t.stop()

    # ---- connecting

    def connection(self, h: dict[str, Any], timeout: float) -> tuple[http.client.HTTPConnection, str]:
        """An HTTP connection to the host (a pinned TLS one, or one through the tunnel) and the Host header to use."""
        if h["kind"] == "ssh":
            tunnel = self.tunnel(h)
            if not tunnel.wait_ready(15):
                raise OSError(tunnel.error or "The SSH tunnel is not up yet")
            local = f"127.0.0.1:{tunnel.local_port}"
            return http.client.HTTPConnection("127.0.0.1", tunnel.local_port, timeout=timeout), local
        ip = _resolve(h["address_host"])
        port = int(h["port"])
        hostheader = f"[{ip}]:{port}" if ":" in ip else f"{ip}:{port}"
        if h.get("scheme") == "http":
            return http.client.HTTPConnection(ip, port, timeout=timeout), hostheader
        return PinnedHTTPSConnection(ip, port, h.get("fingerprint", ""), timeout), hostheader

    def request(
        self, h: dict[str, Any], method: str, path: str, body: Any = None, timeout: float = 15, token: bool = True
    ) -> tuple[int, Any]:
        """One JSON request to the host, for the add and check dialogs."""
        conn, hostheader = self.connection(h, timeout)
        headers = {"Host": hostheader, "Accept": "application/json"}
        if token and h.get("token"):
            headers["Authorization"] = "Bearer " + h["token"]
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        finally:
            conn.close()
        try:
            return resp.status, json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            return resp.status, None


def describe_failure(exc: BaseException) -> tuple[str, str, dict[str, Any]]:
    """(code, message, extra) for what went wrong when talking to a host."""
    if isinstance(exc, FingerprintError):
        return (
            "fingerprint_changed",
            "The host's certificate has changed since you paired. It may be a new installation of the host, or "
            "somebody in between. Nothing was sent.",
            {"seen": tlsgen.pretty(exc.seen), "pinned": tlsgen.pretty(exc.expected)},
        )
    if isinstance(exc, ssl.SSLError):
        return "tls", f"Could not set up an encrypted connection: {exc.reason or exc}", {}
    if isinstance(exc, socket.timeout | TimeoutError):
        return "unreachable", "The host did not answer in time.", {}
    return "unreachable", f"Host not reachable: {exc}", {}


# ------------------------------------------------------------------------------------------------ the service


class Remote:
    def __init__(self, studio: Any) -> None:
        self.studio = studio
        self.clients = Clients(studio.data_dir)
        self.hosts = Hosts(studio.data_dir)


def service(studio: Any) -> Remote:
    found = studio.__dict__.get("_remote")
    if found is None:
        found = studio.__dict__.setdefault("_remote", Remote(studio))
    return found


def shutdown(studio: Any) -> None:
    found = studio.__dict__.get("_remote")
    if found is not None:
        found.hosts.shutdown()


# ------------------------------------------------------------------------------------------------ handler glue


def bearer(handler: Any) -> str:
    auth = handler.headers.get("Authorization", "")
    return auth[7:].strip() if auth.startswith("Bearer ") else ""


def authenticate(handler: Any) -> bool | None:
    """True: a valid client token (handler.client is set). False: a client token that is not valid (revoked).
    None: no client token at all, the other ways of signing in apply."""
    token = bearer(handler)
    if not token.startswith(TOKEN_PREFIX):
        return None
    client = service(handler.studio).clients.find(token, handler.client_address[0])
    if client is None:
        return False
    handler.client = client
    return True


def enforce(handler: Any, path: str) -> bool:
    """Rights of a paired client. Answers the request itself when it is refused and returns False."""
    refused = check_rights(handler.client, handler.command, path, handler.headers.get("X-FLE-Confirm") == "1")
    if refused is None:
        return True
    status, message, code = refused
    err: dict[str, Any] = {"message": message, "code": code}
    if status == 428:
        err["confirm"] = True
    handler.send_json(status, {"error": err})
    return False


def _fail(handler: Any, exc: RemoteError) -> None:
    err: dict[str, Any] = {"message": exc.message}
    if exc.code:
        err["code"] = exc.code
    handler.send_json(exc.status, {"error": err})


def _body(handler: Any, limit: int = 65536) -> dict[str, Any]:
    return handler.read_json()


# ---- the host's endpoints


def _listen_hints(studio: Any) -> list[str]:
    info = getattr(studio, "listen", None) or {}
    port = info.get("port") or DEFAULT_PORT
    names = [n for n in tlsgen._local_names() if n not in ("localhost", "127.0.0.1", "::1") and ":" not in n]
    return [f"{n}:{port}" for n in names]


def api_remote(self: Any, query: dict[str, str]) -> None:
    studio = self.studio
    r = service(studio)
    info = getattr(studio, "listen", None) or {}
    self.send_json(
        200,
        {
            "name": host_name(studio),
            "listening": {"host": info.get("host", ""), "port": info.get("port", 0), "tls": bool(info.get("tls"))},
            "fingerprint": getattr(studio, "tls_fingerprint", ""),
            "fingerprint_pretty": tlsgen.pretty(getattr(studio, "tls_fingerprint", "") or ""),
            "addresses": _listen_hints(studio),
            "code": r.clients.code_state(),
            "clients": r.clients.list(),
        },
    )


def api_remote_code(self: Any, query: dict[str, str]) -> None:
    self.read_json()
    self.send_json(200, service(self.studio).clients.new_code())


def api_remote_pair(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    try:
        result = service(self.studio).clients.pair(
            str(body.get("proof") or ""),
            str(body.get("name") or ""),
            self.client_address[0],
            getattr(self.studio, "tls_fingerprint", "") or "",
        )
    except RemoteError as exc:
        _fail(self, exc)
        return
    self.send_json(
        200, {**result, "host": host_name(self.studio), "fingerprint": getattr(self.studio, "tls_fingerprint", "")}
    )


def api_remote_whoami(self: Any, query: dict[str, str]) -> None:
    client = self.client
    self.send_json(
        200,
        {
            "host": host_name(self.studio),
            "name": (client or {}).get("name", ""),
            "admin": bool((client or {}).get("admin", True)),
            "paired": client is not None,
        },
    )


def api_remote_revoke(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    self.read_json()
    if not service(self.studio).clients.revoke(id):
        raise_not_found(self)
        return
    self.send_json(200, {"ok": True, "clients": service(self.studio).clients.list()})


def api_remote_admin(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    body = self.read_json()
    if not service(self.studio).clients.set_admin(id, bool(body.get("admin"))):
        raise_not_found(self)
        return
    self.send_json(200, {"ok": True, "clients": service(self.studio).clients.list()})


def raise_not_found(handler: Any) -> None:
    handler.send_json(404, {"error": {"message": "No such client"}})


# ---- the client's endpoints


def _probe(hosts: Hosts, h: dict[str, Any]) -> dict[str, Any]:
    try:
        status, data = hosts.request(h, "GET", "/api/remote/whoami", timeout=6)
    except (OSError, http.client.HTTPException) as exc:
        code, message, extra = describe_failure(exc)
        return {"ok": False, "code": code, "message": message, **extra}
    if status == 401:
        return {"ok": False, "code": "unpaired", "message": "The host no longer knows this client. Pair again."}
    if status != 200 or not isinstance(data, dict):
        return {"ok": False, "code": "unreachable", "message": f"The host answered {status}."}
    return {
        "ok": True,
        "host_name": data.get("host", ""),
        "admin": bool(data.get("admin")),
        "name": data.get("name", ""),
    }


def _pair(hosts: Hosts, record: dict[str, Any], code: str, client_name: str) -> dict[str, Any]:
    """Pin the certificate the host shows now and trade the code for a token; returns the changed record."""
    if record["kind"] == "direct" and record["scheme"] == "https":
        try:
            conn, _ = hosts.connection({**record, "fingerprint": ""}, 10)
            conn.connect()  # type: ignore[attr-defined]
            seen = conn.seen  # type: ignore[attr-defined]
            conn.close()
        except (OSError, http.client.HTTPException) as exc:
            raise RemoteError(502, describe_failure(exc)[1], "unreachable") from None
        expected = tlsgen.normalize(record.get("expect") or "")
        if expected and expected != seen:
            raise RemoteError(
                400, "The host shows a different certificate than the fingerprint you entered.", "fingerprint_mismatch"
            )
        record = {**record, "fingerprint": seen}
    else:
        record = {**record, "fingerprint": ""}
    proof = pair_proof(code, record["fingerprint"])
    try:
        status, data = hosts.request(
            record, "POST", "/api/remote/pair", {"proof": proof, "name": client_name}, timeout=15, token=False
        )
    except (OSError, http.client.HTTPException) as exc:
        raise RemoteError(502, describe_failure(exc)[1], "unreachable") from None
    if status != 200 or not isinstance(data, dict) or not data.get("token"):
        err = (data or {}).get("error", {}) if isinstance(data, dict) else {}
        raise RemoteError(
            status if status in (400, 429) else 502,
            err.get("message") or f"The host answered {status}.",
            err.get("code", ""),
        )
    record = {**record, "token": data["token"], "paired": _now()}
    record.pop("expect", None)
    if not record.get("name") or record["name"] == record.get("address"):
        record["name"] = clean_name(data.get("host"), record["name"])
    return record


def api_hosts(self: Any, query: dict[str, str]) -> None:
    hosts = service(self.studio).hosts
    self.send_json(
        200,
        {
            "hosts": [hosts.public(h) for h in hosts.all()],
            "ssh": _ssh_available(),
            "client_name": host_name(self.studio),
        },
    )


def _ssh_available() -> bool:
    try:
        ssh_program()
        return True
    except RemoteError:
        return False


def api_hosts_add(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    r = service(self.studio)
    try:
        kind = str(body.get("kind") or "direct")
        code = str(body.get("code") or "")
        if not normalize_code(code):
            raise RemoteError(400, "Enter the pairing code that the host shows.")
        record: dict[str, Any] = {
            "id": secrets.token_hex(6),
            "kind": kind,
            "color": r.hosts.next_color(),
            "mirror": bool(body.get("mirror")),
            "added": _now(),
            "token": "",
            "fingerprint": "",
        }
        if kind == "ssh":
            target, port, remote_port = check_ssh(body.get("target"), body.get("port"), body.get("remote_port"))
            record.update(
                ssh={"target": target, "port": port, "remote_port": remote_port},
                scheme="http",
                address=target,
                address_host="127.0.0.1",
                port=remote_port,
                name=clean_name(body.get("name"), target.split("@")[-1]),
            )
        elif kind == "direct":
            scheme, name, port = parse_address(str(body.get("address") or ""))
            record.update(
                scheme=scheme,
                address=f"{name}:{port}",
                address_host=name,
                port=port,
                ssh=None,
                name=clean_name(body.get("name"), name),
                expect=str(body.get("fingerprint") or ""),
            )
        else:
            raise RemoteError(400, "Unknown kind of host.")
        if any(h["address"] == record["address"] and h["kind"] == kind for h in r.hosts.all()):
            raise RemoteError(409, "This host is already in the list. Remove it first to pair again.")
        paired = _pair(r.hosts, record, code, host_name(self.studio))
    except RemoteError as exc:
        _fail(self, exc)
        return
    r.hosts.save(paired)
    probe = _probe(r.hosts, paired)
    self.send_json(200, {"host": r.hosts.public(paired), "probe": probe})


def api_host_check(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    self.read_json()
    r = service(self.studio)
    try:
        h = r.hosts.get(id)
    except RemoteError as exc:
        _fail(self, exc)
        return
    probe = _probe(r.hosts, h)
    self.send_json(200, {"host": r.hosts.public(h), **probe})


def api_host_connect(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    """'Reconnect': a new ssh tunnel for an SSH host; for the others it is a plain check."""
    self.read_json()
    r = service(self.studio)
    try:
        h = r.hosts.get(id)
    except RemoteError as exc:
        _fail(self, exc)
        return
    if h["kind"] == "ssh":
        r.hosts.tunnel(h).restart()
    probe = _probe(r.hosts, h)
    self.send_json(200, {"host": r.hosts.public(h), **probe})


def api_host_repair(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    """Pair again with a new code: after a revoked token, or a certificate that really did change."""
    body = self.read_json()
    r = service(self.studio)
    try:
        h = r.hosts.get(id)
        h = {**h, "expect": str(body.get("fingerprint") or "")}
        paired = _pair(r.hosts, h, str(body.get("code") or ""), host_name(self.studio))
    except RemoteError as exc:
        _fail(self, exc)
        return
    r.hosts.save(paired)
    self.send_json(200, {"host": r.hosts.public(paired), "probe": _probe(r.hosts, paired)})


def api_host_update(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    body = self.read_json()
    r = service(self.studio)
    try:
        h = r.hosts.get(id)
    except RemoteError as exc:
        _fail(self, exc)
        return
    if "name" in body:
        h["name"] = clean_name(body["name"], h["name"])
    if "mirror" in body:
        h["mirror"] = bool(body["mirror"])
    if re.match(r"^#[0-9a-fA-F]{6}$", str(body.get("color") or "")):
        h["color"] = body["color"]
    r.hosts.save(h)
    self.send_json(200, {"host": r.hosts.public(h)})


def api_host_remove(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    self.read_json()
    r = service(self.studio)
    r.hosts.remove(id)
    self.send_json(200, {"ok": True, "hosts": [r.hosts.public(h) for h in r.hosts.all()]})


ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"^/api/remote$", "remote"),
    ("POST", r"^/api/remote/code$", "remote_code"),
    ("POST", r"^/api/remote/pair$", "remote_pair"),
    ("GET", r"^/api/remote/whoami$", "remote_whoami"),
    ("POST", r"^/api/remote/clients/(?P<id>[0-9a-f]{12})/revoke$", "remote_revoke"),
    ("POST", r"^/api/remote/clients/(?P<id>[0-9a-f]{12})/admin$", "remote_admin"),
    ("GET", r"^/api/hosts$", "hosts"),
    ("POST", r"^/api/hosts$", "hosts_add"),
    ("POST", r"^/api/hosts/(?P<id>[0-9a-f]{12})/check$", "host_check"),
    ("POST", r"^/api/hosts/(?P<id>[0-9a-f]{12})/connect$", "host_connect"),
    ("POST", r"^/api/hosts/(?P<id>[0-9a-f]{12})/repair$", "host_repair"),
    ("POST", r"^/api/hosts/(?P<id>[0-9a-f]{12})/update$", "host_update"),
    ("POST", r"^/api/hosts/(?P<id>[0-9a-f]{12})/remove$", "host_remove"),
]


def install(handler: type, routes: list[Any], api_error: type) -> None:
    """Add the handlers to the server's Handler class and the routes to its route table."""
    for method, pattern, name in ROUTES:
        routes.append((method, re.compile(pattern), name))
        setattr(handler, f"api_{name}", globals()[f"api_{name}"])


# ------------------------------------------------------------------------------------------------ the pass-through

_FORWARD_REQUEST = ("Content-Type", "Accept", "X-FLE-Confirm")
# The Content-Security-Policy of an MCP App page (mcp_api.py) must survive the trip: it keeps the app in its box.
_FORWARD_RESPONSE = (
    "Content-Type",
    "Content-Disposition",
    "Content-Range",
    "Accept-Ranges",
    "Content-Security-Policy",
    "Permissions-Policy",
)
_CHAT_LIST = re.compile(r"^/api/chats$")
_CHAT_ONE = re.compile(r"^/api/chats/([0-9a-f]{16})$")


def _upload(handler: Any, length: int) -> Iterator[bytes]:
    left = length
    handler._unread = 0
    while left > 0:
        chunk = handler.rfile.read(min(65536, left))
        if not chunk:
            return
        left -= len(chunk)
        yield chunk


def _mirror_dir(handler: Any, host_id: str) -> Path:
    return paths.ensure(handler.studio.data_dir / "mirror" / host_id)


def _mirror_save(handler: Any, h: dict[str, Any], path: str, raw: bytes) -> None:
    if not h.get("mirror"):
        return
    try:
        data = json.loads(raw.decode("utf-8"))
        folder = _mirror_dir(handler, h["id"])
        if _CHAT_LIST.match(path):
            _write(folder / "chats.json", data)
        else:
            m = _CHAT_ONE.match(path)
            if m and isinstance(data, dict):
                _write(folder / f"chat-{m.group(1)}.json", data)
    except (OSError, ValueError):
        pass


def _mirror_serve(handler: Any, h: dict[str, Any], path: str) -> bool:
    """With the host away, a mirrored host still shows its chats (read only)."""
    if not h.get("mirror") or handler.command != "GET":
        return False
    folder = handler.studio.data_dir / "mirror" / h["id"]
    if _CHAT_LIST.match(path):
        data = _read(folder / "chats.json", None)
    else:
        m = _CHAT_ONE.match(path)
        data = _read(folder / f"chat-{m.group(1)}.json", None) if m else None
    if data is None:
        return False
    if isinstance(data, dict):
        data = {**data, "mirror": True}
    handler.send_json(200, data)
    return True


_BUFFER_MAX = 8 * 1024 * 1024  # larger answers are passed on as they arrive instead of being read whole


def _never_forwarded(rest: str) -> bool:
    """Paths that stay on this machine: pairing and host management (`/api/remote`, `/api/remote/...`, `/api/hosts...`), and anything that tries to
    reach them around the check with dot segments (`/api/../api/remote/...`, percent-encoded or not)."""
    low = rest.lower()
    if "%2e" in low or "%2f" in low or "\\" in rest or any(seg in (".", "..") for seg in rest.split("?")[0].split("/")):
        return True
    return rest == "/api/remote" or rest.startswith(("/api/remote/", "/api/hosts"))


def forward(handler: Any, url: Any) -> None:
    """`/h/<id>/api/...` and `/h/<id>/v1/...` to the host, with the client token. The answer is passed on as it
    arrives, so streaming chat, downloads and uploads work the same as on this machine."""
    m = re.match(r"^/h/(?P<id>[0-9a-f]{12})(?P<rest>/.*)$", url.path)
    if not m:
        handler.send_json(404, {"error": {"message": "Not found"}})
        return
    r = service(handler.studio)
    rest = m.group("rest")
    try:
        h = r.hosts.get(m.group("id"))
    except RemoteError as exc:
        _fail(handler, exc)
        return
    app_page = rest.startswith("/app-frame/") and handler.command == "GET"
    if not (rest.startswith(("/api/", "/v1/")) or app_page) or _never_forwarded(rest):
        handler.send_json(403, {"error": {"message": "This path is not passed on to a host."}})
        return
    if not h.get("token"):
        handler.send_json(424, {"error": {"message": "This host is not paired.", "code": "unpaired"}})
        return
    target = rest + (("?" + url.query) if url.query else "")
    stream_like = rest.startswith("/v1/")
    length = handler._unread
    conn: http.client.HTTPConnection | None = None
    try:
        conn, hostheader = r.hosts.connection(h, 900 if stream_like else 60)
        headers = {"Host": hostheader, "Authorization": "Bearer " + h["token"]}
        for name in _FORWARD_REQUEST:
            if handler.headers.get(name):
                headers[name] = handler.headers[name]
        body: Any = None
        if handler.command in ("POST", "PUT", "PATCH") or length:
            headers["Content-Length"] = str(length)
            body = _upload(handler, length) if length else b""
        conn.request(handler.command, target, body=body, headers=headers)
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException) as exc:
        if conn is not None:
            conn.close()
        if length:
            handler.close_connection = True  # part of the upload may be unread
        code, message, extra = describe_failure(exc)
        if code == "unreachable" and _mirror_serve(handler, h, rest):
            return
        handler.send_json(502, {"error": {"message": message, "code": code, **extra}})
        return
    if resp.status == 401:
        resp.read()
        conn.close()
        handler.send_json(
            424,
            {
                "error": {
                    "message": "The host does not accept this client any more. It may have been revoked: pair again.",
                    "code": "unpaired",
                }
            },
        )
        return
    ctype = resp.getheader("Content-Type", "application/octet-stream")
    size = resp.getheader("Content-Length")
    sse = "text/event-stream" in ctype
    # Small known-length answers are read whole (and mirrored); large ones (an export, a download) are streamed so they never sit in memory.
    if size is not None and not sse and size.isdigit() and int(size) <= _BUFFER_MAX:
        data = resp.read()
        conn.close()
        if resp.status == 200:
            _mirror_save(handler, h, rest, data)
        extra = {n: resp.getheader(n) for n in _FORWARD_RESPONSE if n != "Content-Type" and resp.getheader(n)}
        handler._send(resp.status, data, ctype, {k: v for k, v in extra.items() if v})
        return
    handler.send_response(resp.status)
    handler.send_header("Content-Type", ctype)
    for n in _FORWARD_RESPONSE[1:]:
        if resp.getheader(n):
            handler.send_header(n, resp.getheader(n))
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.close_connection = True
    try:
        while True:
            chunk = resp.read1(65536)
            if not chunk:
                break
            handler.wfile.write(chunk)
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass  # the page stopped reading; closing the host side ends the generation there too
    finally:
        conn.close()
