"""Downloads._fetch against a local HTTP server: fresh download, resume, no-Range server, checksum, errors, retry."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from fle_studio.downloads import Downloads

DATA = bytes(range(256)) * 4096  # 1 MiB


class _Server:
    """Serves DATA at /file. Modes: range (honours Range), norange (always 200), cut (first reply is cut short)."""

    def __init__(self, mode: str = "range") -> None:
        self.mode = mode
        self.requests: list[str] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802
                outer.requests.append(self.headers.get("Range", ""))
                if self.path == "/missing":
                    self.send_error(404)
                    return
                if self.path == "/gated":
                    self.send_error(403)
                    return
                start = 0
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes=") and outer.mode != "norange":
                    start = int(rng[6:].split("-")[0])
                    if start >= len(DATA):
                        self.send_response(416)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                else:
                    self.send_response(200)
                body = DATA[start:]
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if outer.mode == "cut" and len(outer.requests) == 1:
                    self.wfile.write(body[: len(body) // 3])  # connection closes with bytes missing
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def served(request: pytest.FixtureRequest):
    srv = _Server(getattr(request, "param", "range"))
    yield srv
    srv.close()


def _job(
    url: str, dest: Path, *, size: int = len(DATA), sha: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    f = {"url": url, "dest": str(dest), "size": size, "sha256": sha, "done": 0}
    return {"files": [f], "state": "queued", "speed": 0.0, "error": ""}, f


def _dl(tmp_path: Path) -> Downloads:
    return Downloads(tmp_path / "data", lambda: "")


def test_fresh_download_with_checksum(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "m" / "a.gguf"
    job, f = _job(served.url + "/file", dest, sha=hashlib.sha256(DATA).hexdigest())
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA
    assert f["done"] == len(DATA)
    assert not dest.with_name("a.gguf.part").exists()


def test_unknown_size_is_learned_from_the_server(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    job, f = _job(served.url + "/file", dest, size=0)
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA
    assert f["size"] == len(DATA)


def test_resume_continues_from_the_part_file_and_verifies_it(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    dest.with_name("a.gguf.part").write_bytes(DATA[:300_000])
    job, f = _job(served.url + "/file", dest, sha=hashlib.sha256(DATA).hexdigest())
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA
    assert served.requests == ["bytes=300000-"]


@pytest.mark.parametrize("served", ["norange"], indirect=True)
def test_server_without_range_restarts_the_file(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    dest.with_name("a.gguf.part").write_bytes(DATA[:1000])
    job, f = _job(served.url + "/file", dest, sha=hashlib.sha256(DATA).hexdigest())
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA  # one copy, not the old 1000 bytes plus a whole new file


def test_part_file_larger_than_the_file_is_dropped(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    dest.with_name("a.gguf.part").write_bytes(DATA + b"junk")
    job, f = _job(served.url + "/file", dest)
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA


def test_complete_part_file_answered_with_416_is_accepted(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    dest.with_name("a.gguf.part").write_bytes(DATA)
    job, f = _job(served.url + "/file", dest, sha=hashlib.sha256(DATA).hexdigest())
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA


def test_existing_file_of_the_right_size_is_not_fetched_again(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    dest.write_bytes(DATA)
    job, f = _job(served.url + "/file", dest)
    _dl(tmp_path)._fetch(job, f)
    assert served.requests == []
    assert f["done"] == len(DATA)


def test_checksum_mismatch_moves_the_file_aside(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    job, f = _job(served.url + "/file", dest, sha="0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _dl(tmp_path)._fetch(job, f)
    assert not dest.exists()
    assert dest.with_name("a.gguf.bad").read_bytes() == DATA
    assert f["done"] == 0


def test_404_and_403_become_readable_errors(served: _Server, tmp_path: Path) -> None:
    job, f = _job(served.url + "/missing", tmp_path / "a.gguf")
    with pytest.raises(RuntimeError, match="Not found"):
        _dl(tmp_path)._fetch(job, f)
    job, f = _job(served.url + "/gated", tmp_path / "b.gguf")
    with pytest.raises(RuntimeError, match="Access denied"):
        _dl(tmp_path)._fetch(job, f)


@pytest.mark.parametrize("served", ["cut"], indirect=True)
def test_connection_cut_mid_file_is_retried_from_where_it_stopped(served: _Server, tmp_path: Path) -> None:
    dest = tmp_path / "a.gguf"
    job, f = _job(served.url + "/file", dest, sha=hashlib.sha256(DATA).hexdigest())
    _dl(tmp_path)._fetch(job, f)
    assert dest.read_bytes() == DATA
    assert len(served.requests) == 2 and served.requests[1].startswith("bytes=")
