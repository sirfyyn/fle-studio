"""A stand-in for the part of api.github.com that the update check asks, on a loopback port.

    with FakeGitHub(releases=[release("v0.2.0", "Notes")]) as gh:
        os.environ["FLE_STUDIO_GITHUB_API"] = gh.url
        ...
        gh.requests   # [(path, headers), ...] of everything that was asked

`status` makes every answer an error with that HTTP status. The tests need no network and never reach the real
GitHub.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

STUDIO = "/repos/sirfyyn/fle-studio"
ENGINE = "/repos/sirfyyn/FastLocalExperts"


def release(
    tag: str, body: str = "", prerelease: bool = False, draft: bool = False, assets: list[str] | None = None
) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "name": tag,
        "body": body,
        "prerelease": prerelease,
        "draft": draft,
        "published_at": "2026-10-11T10:00:00Z",
        "html_url": "https://evil.example/not-used",  # the app must build its own link
        "assets": [
            {"name": n, "browser_download_url": f"https://example.invalid/{n}", "size": 1} for n in assets or []
        ],
    }


class FakeGitHub:
    def __init__(
        self, releases: list[dict[str, Any]] | None = None, engine: dict[str, Any] | None = None, status: int = 0
    ) -> None:
        self.releases = releases or []
        self.engine = engine
        self.status = status
        self.requests: list[tuple[str, dict[str, str]]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                owner.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
                if owner.status:
                    self._send(owner.status, {"message": "error"})
                    return
                path = self.path.split("?")[0]
                if path == f"{STUDIO}/releases/latest":
                    stable = [r for r in owner.releases if not r.get("prerelease") and not r.get("draft")]
                    self._send(200, stable[0]) if stable else self._send(404, {"message": "Not Found"})
                elif path == f"{STUDIO}/releases":
                    self._send(200, owner.releases)
                elif path == f"{ENGINE}/releases/latest" and owner.engine is not None:
                    self._send(200, owner.engine)
                else:
                    self._send(404, {"message": "Not Found"})

            def _send(self, status: int, data: Any) -> None:
                raw = json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> FakeGitHub:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def paths(self) -> list[str]:
        return [p for p, _ in self.requests]
