"""The model switcher client (switch.py): lists the models of a switcher, switches by hand, shows progress and errors."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from fle_studio.switch import Switch


class Fake:
    def __init__(self) -> None:
        self.loaded = ["rhea"]
        self.busy = False
        self.target = None
        self.delay = 0.0
        self.answer = (200, {"ok": True})
        self.posts: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # noqa: D401
                pass

            def _send(self, code, data):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"familien": {"rhea": {"alias": "rhea"}, "katy": {"alias": "katy"}}, "geladen": fake.loaded,
                                 "wechsel_laeuft": fake.busy, "wechsel_ziel": fake.target})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                fake.posts.append(json.loads(self.rfile.read(n)))
                fake.busy, fake.target = True, fake.posts[-1]["ziel"]
                time.sleep(fake.delay)
                code, data = fake.answer
                if code == 200:
                    fake.loaded = [fake.target]
                fake.busy = False
                self._send(code, data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.srv.shutdown()


@pytest.fixture()
def fake():
    f = Fake()
    yield f
    f.close()


def wait(sw, until, s=5.0):
    t = time.time()
    while time.time() - t < s:
        st = sw.status()
        if until(st):
            return st
        time.sleep(0.05)
    raise AssertionError(sw.status())


def test_without_url_nothing_is_available(monkeypatch):
    monkeypatch.delenv("FLE_STUDIO_SWITCH_URL", raising=False)
    assert Switch(lambda: "").status() == {"available": False}


def test_lists_models_and_marks_the_loaded_one(fake):
    st = Switch(lambda: fake.url).status()
    assert st["available"] and st["active"] == "rhea" and not st["busy"]
    assert [(t["id"], t["active"]) for t in st["targets"]] == [("rhea", True), ("katy", False)]


def test_switch_runs_in_the_background_and_ends_loaded(fake):
    fake.delay = 0.3
    sw = Switch(lambda: fake.url)
    st = sw.start("katy")
    assert st["busy"] and st["target"] == "katy"
    st = wait(sw, lambda s: not s["busy"])
    assert st["active"] == "katy" and not st["error"] and fake.posts == [{"ziel": "katy", "grund": "FLE Studio"}]


def test_switching_to_the_loaded_model_does_nothing(fake):
    sw = Switch(lambda: fake.url)
    assert sw.start("rhea")["active"] == "rhea" and fake.posts == []


def test_unknown_model_is_refused(fake):
    with pytest.raises(ValueError):
        Switch(lambda: fake.url).start("zeta")


def test_a_refused_switch_shows_the_reason(fake):
    fake.answer = (503, {"ok": False, "fehler": "Messmarke /run/fyos-messung-laeuft"})
    sw = Switch(lambda: fake.url)
    sw.start("katy")
    st = wait(sw, lambda s: not s["busy"])
    assert st["active"] == "rhea" and "Messmarke" in st["error"]


def test_second_switch_while_one_runs_is_refused(fake):
    fake.delay = 0.5
    sw = Switch(lambda: fake.url)
    sw.start("katy")
    with pytest.raises(RuntimeError):
        sw.start("rhea")
    wait(sw, lambda s: not s["busy"])


def test_unreachable_switcher_reports_it():
    st = Switch(lambda: "http://127.0.0.1:9").status()
    assert not st["available"] and "not reachable" in st["error"]
