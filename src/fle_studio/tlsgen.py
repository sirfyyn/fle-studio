"""A self-signed TLS certificate for host mode, made with the standard library alone.

FLE Studio has no dependencies, and Windows has no `openssl` on the PATH, so the certificate is built here: an ECDSA
P-256 key, a minimal X.509 v3 certificate (DER) signed with SHA-256, both written as PEM. The key is only used to sign
this one certificate and to answer TLS handshakes. Clients do not trust it through a certificate authority; they pin
its SHA-256 fingerprint when they pair (trust on first use), see remote.py.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import ipaddress
import os
import secrets
import socket
import ssl
import threading
from pathlib import Path

from . import paths

# NIST P-256 (secp256r1)
_P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
_A = _P - 3
_B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
_G = (
    0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
    0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
)
_N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

Point = tuple[int, int] | None


def _add(p: Point, q: Point) -> Point:
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2:
        if (y1 + y2) % _P == 0:
            return None
        lam = (3 * x1 * x1 + _A) * pow(2 * y1, -1, _P) % _P
    else:
        lam = (y2 - y1) * pow(x2 - x1, -1, _P) % _P
    x3 = (lam * lam - x1 - x2) % _P
    return x3, (lam * (x1 - x3) - y1) % _P


def _mul(k: int, p: Point) -> Point:
    out: Point = None
    while k:
        if k & 1:
            out = _add(out, p)
        p = _add(p, p)
        k >>= 1
    return out


def _sign(d: int, digest: bytes) -> tuple[int, int]:
    z = int.from_bytes(digest, "big")
    while True:
        k = secrets.randbelow(_N - 1) + 1
        point = _mul(k, _G)
        assert point is not None
        r = point[0] % _N
        if r == 0:
            continue
        s = pow(k, -1, _N) * (z + r * d) % _N
        if s:
            return r, s


# ------------------------------------------------------------------------------------------------ DER


def _len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def _tlv(tag: int, body: bytes) -> bytes:
    return bytes([tag]) + _len(len(body)) + body


def _seq(*parts: bytes) -> bytes:
    return _tlv(0x30, b"".join(parts))


def _int(n: int) -> bytes:
    raw = n.to_bytes(max(1, (n.bit_length() + 8) // 8), "big")
    return _tlv(0x02, raw)


def _oid(dotted: str) -> bytes:
    nums = [int(x) for x in dotted.split(".")]
    out = bytes([nums[0] * 40 + nums[1]])
    for n in nums[2:]:
        chunk = [n & 0x7F]
        n >>= 7
        while n:
            chunk.append(0x80 | (n & 0x7F))
            n >>= 7
        out += bytes(reversed(chunk))
    return _tlv(0x06, out)


def _bits(data: bytes) -> bytes:
    return _tlv(0x03, b"\x00" + data)


def _time(when: dt.datetime) -> bytes:
    return _tlv(0x17, when.strftime("%y%m%d%H%M%SZ").encode())


OID_EC = _oid("1.2.840.10045.2.1")
OID_P256 = _oid("1.2.840.10045.3.1.7")
OID_ECDSA_SHA256 = _oid("1.2.840.10045.4.3.2")
SIG_ALG = _seq(OID_ECDSA_SHA256)


def _name(common_name: str) -> bytes:
    return _seq(_tlv(0x31, _seq(_oid("2.5.4.3"), _tlv(0x0C, common_name.encode("utf-8")))))


def _ext(oid: str, value: bytes, critical: bool = False) -> bytes:
    return _seq(_oid(oid), *([_tlv(0x01, b"\xff")] if critical else []), _tlv(0x04, value))


def _san(names: list[str]) -> bytes:
    items = []
    for name in names:
        try:
            items.append(_tlv(0x87, ipaddress.ip_address(name).packed))
        except ValueError:
            items.append(_tlv(0x82, name.encode("ascii", "ignore")))
    return _seq(*items)


def make_certificate(common_name: str, names: list[str], years: int = 10) -> tuple[bytes, str]:
    """Return (certificate as DER, private key as PEM text)."""
    d = secrets.randbelow(_N - 1) + 1
    pub = _mul(d, _G)
    assert pub is not None
    point = b"\x04" + pub[0].to_bytes(32, "big") + pub[1].to_bytes(32, "big")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    extensions = _seq(
        _ext("2.5.29.19", _seq(), critical=True),  # basicConstraints: not a CA
        _ext("2.5.29.15", _tlv(0x03, b"\x07\x80"), critical=True),  # keyUsage: digitalSignature
        _ext("2.5.29.37", _seq(_oid("1.3.6.1.5.5.7.3.1"))),  # extendedKeyUsage: serverAuth
        _ext("2.5.29.17", _san(names)),
    )
    serial = int.from_bytes(secrets.token_bytes(16), "big") >> 1 or 1
    tbs = _seq(
        _tlv(0xA0, _int(2)),
        _int(serial),
        SIG_ALG,
        _name(common_name),
        _seq(_time(now - dt.timedelta(days=1)), _time(now + dt.timedelta(days=365 * years))),
        _name(common_name),
        _seq(_seq(OID_EC, OID_P256), _bits(point)),
        _tlv(0xA3, extensions),
    )
    r, s = _sign(d, hashlib.sha256(tbs).digest())
    der = _seq(tbs, SIG_ALG, _bits(_seq(_int(r), _int(s))))
    key = _seq(_int(1), _tlv(0x04, d.to_bytes(32, "big")), _tlv(0xA0, OID_P256), _tlv(0xA1, _bits(point)))
    return der, _pem("EC PRIVATE KEY", key)


def _pem(label: str, der: bytes) -> str:
    body = base64.b64encode(der).decode()
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"


# ------------------------------------------------------------------------------------------------ the host's files


def fingerprint_of(der: bytes) -> str:
    """SHA-256 of the certificate, as 64 lower-case hex digits."""
    return hashlib.sha256(der).hexdigest()


def pretty(fp: str) -> str:
    """AA:BB:CC style, which is how people compare fingerprints."""
    fp = fp.lower().replace(":", "")
    return ":".join(fp[i : i + 2] for i in range(0, len(fp), 2)).upper()


def normalize(fp: str) -> str:
    return "".join(c for c in fp.lower() if c in "0123456789abcdef")


RESOLVE_TIMEOUT = 2.0


def _route_addresses() -> set[str]:
    """The addresses this machine sends from, found without DNS (a UDP connect sends nothing)."""
    found = set()
    for family, probe in ((socket.AF_INET, "192.0.2.1"), (socket.AF_INET6, "2001:db8::1")):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((probe, 9))
                found.add(str(s.getsockname()[0]).split("%", 1)[0])
        except OSError:
            continue
    return found


def _resolved_addresses(host: str) -> set[str]:
    """The addresses the host name resolves to, or nothing when the lookup takes too long.

    An unresolvable own host name can block for many seconds (seen on macOS), so the lookup runs aside with a limit.
    """
    found: set[str] = set()

    def look() -> None:
        try:
            for info in socket.getaddrinfo(host, None):
                found.add(str(info[4][0]).split("%", 1)[0])
        except OSError:
            pass

    worker = threading.Thread(target=look, daemon=True)
    worker.start()
    worker.join(RESOLVE_TIMEOUT)
    return set(found) if not worker.is_alive() else set()


def _local_names() -> list[str]:
    names = {"localhost", "127.0.0.1", "::1"}
    host = socket.gethostname()
    names.add(host)
    names |= _route_addresses()
    names |= _resolved_addresses(host)
    return sorted(n for n in names if n and n.isascii())


def ensure(data_dir: Path, name: str = "") -> tuple[Path, Path, str]:
    """The certificate and key files in `<data>/tls`, made on first use. Returns (cert, key, fingerprint)."""
    folder = paths.ensure(data_dir / "tls")
    cert, key = folder / "host.crt", folder / "host.key"
    if cert.exists() and key.exists():
        try:
            return cert, key, fingerprint_of(ssl.PEM_cert_to_DER_cert(cert.read_text(encoding="ascii")))
        except (OSError, ValueError):
            pass  # unreadable: make a new one (clients will see the change and must pair again)
    der, key_pem = make_certificate(f"FLE Studio {name or socket.gethostname()}", _local_names())
    key.write_text(key_pem, encoding="ascii")
    if not paths.IS_WIN:
        os.chmod(key, 0o600)
    cert.write_text(_pem("CERTIFICATE", der), encoding="ascii")
    return cert, key, fingerprint_of(der)


def server_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    return ctx
