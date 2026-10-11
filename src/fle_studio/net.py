"""HTTP helpers on urllib: one TLS context that works with python.org, uv and system Pythons alike, and a token
that is only ever sent to Hugging Face itself (never along a redirect to a CDN)."""

from __future__ import annotations

import json
import os
import ssl
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__

USER_AGENT = f"fle-studio/{__version__} (+https://github.com/sirfyyn/fle-studio)"
HF_HOSTS = ("huggingface.co", "hf.co")
_CA_FILES = (
    "/etc/ssl/cert.pem",
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/usr/local/etc/openssl/cert.pem",
)


@lru_cache(maxsize=1)
def ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    found = ssl.get_default_verify_paths()
    has_default = bool(
        (found.cafile and os.path.exists(found.cafile))
        or (found.capath and os.path.isdir(found.capath) and os.listdir(found.capath))
    )
    if not has_default and not os.environ.get("SSL_CERT_FILE"):
        # python.org builds on macOS ship without CA certificates until "Install Certificates" runs.
        for cand in _CA_FILES:
            if os.path.exists(cand):
                ctx.load_verify_locations(cand)
                break
    try:
        import certifi  # noqa: PLC0415 - optional, used when present

        ctx.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return ctx


def open_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    token: str = "",
    timeout: float = 60,
    method: str = "GET",
    data: bytes | None = None,
) -> Any:
    req = Request(url, data=data, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    host = (urlparse(url).hostname or "").lower()
    if token and (host in HF_HOSTS or host.endswith(".huggingface.co")):
        req.add_unredirected_header("Authorization", f"Bearer {token}")  # not copied onto redirects
    return urlopen(req, timeout=timeout, context=ssl_context())


def get_json(
    url: str, *, token: str = "", timeout: float = 30, headers: dict[str, str] | None = None
) -> tuple[Any, Any]:
    with open_url(url, token=token, timeout=timeout, headers=headers) as r:
        return json.loads(r.read().decode("utf-8")), r.headers
