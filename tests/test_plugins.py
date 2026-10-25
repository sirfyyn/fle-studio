"""Plugins (plugins.py): finding, turning on and off, routes, static files and the rights of paired clients."""

from __future__ import annotations

import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fle_studio import plugins, remote
from fle_studio.server import Studio, make_server

# ------------------------------------------------------------------------------------------------ a stand-in plugin


def demo_register(web: Path, calls: list[str]) -> Any:
    def register(ctx: plugins.PluginContext) -> None:
        calls.append(f"register {ctx.id} api {ctx.api_version}")
        ctx.page("Demo", icon="spark")
        ctx.web(web)

        def hello(req: plugins.Request) -> tuple[int, Any]:
            return 200, {"hello": req.query.get("name", "world"), "data_dir": str(ctx.data_dir)}

        def echo(req: plugins.Request) -> tuple[int, Any]:
            return 201, {"item": req.params["item"], "body": req.json()}

        ctx.route("GET", r"^/hello$", hello)
        ctx.route("POST", r"^/items/(?P<item>[a-z]{1,8})$", echo)
        ctx.on_stop(lambda: calls.append("stop"))

    return register


def broken_register(ctx: plugins.PluginContext) -> None:
    raise RuntimeError("cannot start")


@pytest.fixture
def web(tmp_path: Path) -> Path:
    folder = tmp_path / "pkg" / "web"
    (folder / "sub").mkdir(parents=True)
    (folder / "main.js").write_text("export function mount(main, kit) { return null; }\n")
    (folder / "sub" / "x.css").write_text("body{}\n")
    (tmp_path / "pkg" / "secret.txt").write_text("not for the web\n")
    return folder


# ------------------------------------------------------------------------------------------------ the registry


def test_nothing_is_on_until_turned_on(tmp_path: Path, web: Path) -> None:
    calls: list[str] = []
    reg = plugins.Plugins(tmp_path, "0.0", lambda: [plugins.Found("demo", lambda: demo_register(web, calls), "p", "1")])
    reg.start()
    view = reg.view()
    assert calls == [] and view["plugins"][0]["enabled"] is False and view["plugins"][0]["loaded"] is False
    reg.set_enabled("demo", True)
    assert calls == ["register demo api 1"]
    assert json.loads((tmp_path / "plugins.json").read_text()) == {"enabled": ["demo"]}
    item = reg.view()["plugins"][0]
    assert item["loaded"] and item["page"] == {"title": "Demo", "icon": "spark", "script": "main.js", "path": ""}
    assert (tmp_path / "plugins" / "demo").is_dir()

    again = plugins.Plugins(
        tmp_path, "0.0", lambda: [plugins.Found("demo", lambda: demo_register(web, calls), "p", "1")]
    )
    again.start()  # a new run loads what is on
    assert again.view()["plugins"][0]["loaded"]

    reg.set_enabled("demo", False)
    assert "stop" in calls and reg.view()["plugins"][0]["loaded"] is False
    with pytest.raises(plugins.PluginError):
        reg.handle("GET", "/api/p/demo/hello", {}, dict)


def test_a_broken_plugin_is_reported_and_studio_goes_on(tmp_path: Path) -> None:
    reg = plugins.Plugins(tmp_path, "0.0", lambda: [plugins.Found("bad", lambda: broken_register)])
    reg.start()
    reg.set_enabled("bad", True)
    item = reg.view()["plugins"][0]
    assert item["enabled"] and not item["loaded"] and "cannot start" in item["error"]


def test_unknown_plugins_and_bad_ids(tmp_path: Path) -> None:
    reg = plugins.Plugins(tmp_path, "0.0", lambda: [])
    reg.start()
    with pytest.raises(plugins.PluginError) as err:
        reg.set_enabled("nope", True)
    assert err.value.status == 404
    (tmp_path / "plugins.json").write_text(json.dumps({"enabled": ["ok", "Bad", "../x", 3]}))
    assert reg.enabled() == ["ok"]


def test_context_checks_what_a_plugin_registers(tmp_path: Path) -> None:
    ctx = plugins.PluginContext("x", tmp_path, "0.0")
    for bad in ("../a.js", "a.py", ""):
        with pytest.raises(ValueError):
            ctx.page("T", script=bad)
    with pytest.raises(ValueError):
        ctx.route("PATCH", r"^/a$", lambda r: (200, {}))
    with pytest.raises(ValueError):
        ctx.route("GET", r"/a", lambda r: (200, {}))
    ctx.write_settings({"a": 1})
    assert ctx.read_settings() == {"a": 1}


# ------------------------------------------------------------------------------------------------ bundled plugins


def own_register(calls: list[str]) -> Any:
    def register(ctx: plugins.PluginContext) -> None:
        calls.append(f"register {ctx.id} bundled={ctx.bundled}")
        ctx.describe("Own", "ships with the package")
        ctx.boot("boot.js")
        ctx.page("Own", path="/own")

        def hello(handler: Any, query: dict[str, str], item: str) -> None:
            handler.sent.append((item, query))

        ctx.api("GET", r"^/api/own/(?P<item>[a-z]+)$", hello)
        ctx.on_stop(lambda: calls.append("stop"))

    return register


class Sink:
    def __init__(self) -> None:
        self.sent: list[Any] = []


def test_plugins_of_the_package_itself_are_on_until_turned_off(tmp_path: Path) -> None:
    calls: list[str] = []
    find = lambda: [  # noqa: E731
        plugins.Found("own", lambda: own_register(calls), "fle-studio", "1", bundled=True),
        plugins.Found("other", lambda: own_register([]), "someone-else", "1"),
    ]
    reg = plugins.Plugins(tmp_path, "0.0", find)
    reg.start()
    assert calls == ["register own bundled=True"], "only the bundled plugin starts by itself"
    other, mine = reg.view()["plugins"]  # sorted by id
    assert mine["bundled"] and mine["enabled"] and mine["loaded"] and mine["boot"] == "boot.js"
    assert mine["title"] == "Own" and mine["page"]["path"] == "/own"
    assert not other["bundled"] and not other["enabled"] and not other["loaded"]

    sink = Sink()
    assert reg.handle_raw(sink, "GET", "/api/own/abc", {"q": "1"})
    assert sink.sent == [("abc", {"q": "1"})]
    assert not reg.handle_raw(sink, "POST", "/api/own/abc", {})
    assert not reg.handle_raw(sink, "GET", "/api/elsewhere", {})

    reg.set_enabled("own", False)
    assert "stop" in calls and not reg.view()["plugins"][1]["enabled"]
    assert json.loads((tmp_path / "plugins.json").read_text()) == {"enabled": [], "disabled": ["own"]}
    assert not reg.handle_raw(sink, "GET", "/api/own/abc", {}), "the routes of a plugin that is off do not exist"

    again = plugins.Plugins(tmp_path, "0.0", find)
    again.start()  # the choice is kept over a restart
    assert not again.view()["plugins"][1]["loaded"]
    again.set_enabled("own", True)
    assert again.view()["plugins"][1]["loaded"]
    assert json.loads((tmp_path / "plugins.json").read_text()) == {"enabled": ["own"]}


def test_the_core_runs_without_the_plugins_of_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(plugins.NO_BUNDLED, "1")
    assert not any(f.bundled for f in plugins.discover())


def test_only_the_package_itself_may_take_a_path_or_a_hook(tmp_path: Path) -> None:
    foreign = plugins.PluginContext("x", tmp_path, "0.0")
    with pytest.raises(ValueError):
        foreign.page("T", path="/forge")
    own = plugins.PluginContext("x", tmp_path, "0.0", bundled=True)
    for bad in ("forge", "/a/b", "/Forge", ""):
        if bad:
            with pytest.raises(ValueError):
                own.page("T", path=bad)
    own.page("T", path="/forge")
    with pytest.raises(ValueError):
        own.api("GET", r"^/elsewhere$", lambda h, q: None)
    with pytest.raises(ValueError):
        own.boot("../x.js")
    with pytest.raises(ValueError):
        own.hook("no-such-hook", lambda: None)


# ------------------------------------------------------------------------------------------------ on a real server


class Page:
    def __init__(self, studio: Studio, port: int) -> None:
        self.studio, self.port = studio, port

    def call(self, method: str, path: str, body: Any = None, token: bool = True) -> tuple[int, Any, dict[str, str]]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-FLE-Token"] = self.studio.token
        conn.request(method, path, body=json.dumps(body or {}) if method == "POST" else None, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            data: Any = json.loads(raw or b"null")
        except ValueError:
            data = raw
        return resp.status, data, dict(resp.getheaders())


@pytest.fixture
def page(tmp_path: Path, web: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Page]:
    calls: list[str] = []
    monkeypatch.setattr(
        plugins, "discover", lambda: [plugins.Found("demo", lambda: demo_register(web, calls), "demo-pkg", "1.2")]
    )
    st = Studio(tmp_path / "home")
    server = make_server(st, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    yield Page(st, int(server.server_address[1]))
    server.shutdown()
    server.server_close()
    st.plugins.stop()


def test_routes_and_files_only_while_on(page: Page) -> None:
    status, view, _ = page.call("GET", "/api/plugins")
    assert status == 200 and view["plugins"][0]["package"] == "demo-pkg" and not view["plugins"][0]["loaded"]
    assert page.call("GET", "/api/p/demo/hello")[0] == 404
    assert page.call("GET", "/plugins/demo/main.js")[0] == 404

    assert page.call("POST", "/api/plugins/demo/enable", token=False)[0] == 403  # needs the page token
    status, view, _ = page.call("POST", "/api/plugins/demo/enable")
    assert status == 200 and view["plugins"][0]["loaded"]

    status, data, _ = page.call("GET", "/api/p/demo/hello?name=max")
    assert status == 200 and data["hello"] == "max"
    status, data, _ = page.call("POST", "/api/p/demo/items/abc", {"n": 1})
    assert status == 201 and data == {"item": "abc", "body": {"n": 1}}
    assert page.call("POST", "/api/p/demo/items/abc", {"n": 1}, token=False)[0] == 403
    assert page.call("GET", "/api/p/demo/items/abc")[0] == 405
    assert page.call("GET", "/api/p/demo/nothing")[0] == 404

    status, body, headers = page.call("GET", "/plugins/demo/main.js")
    assert status == 200 and b"export function mount" in body
    assert (
        headers["Content-Type"].startswith("text/javascript")
        and "script-src 'self'" in headers["Content-Security-Policy"]
    )
    assert page.call("GET", "/plugins/demo/sub/x.css")[0] == 200
    assert page.call("GET", "/plugins/demo/../secret.txt")[0] == 404
    assert page.call("GET", "/plugins/demo/%2e%2e/secret.txt")[0] == 404

    status, view, _ = page.call("POST", "/api/plugins/demo/disable")
    assert status == 200 and not view["plugins"][0]["loaded"]
    assert page.call("GET", "/api/p/demo/hello")[0] == 404


def test_the_page_falls_back_to_the_app_for_plugin_paths(page: Page) -> None:
    status, body, headers = page.call("GET", "/p/demo")
    assert status == 200 and b"fle-token" in body and headers["Content-Type"].startswith("text/html")


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/plugins"),
        ("POST", "/api/plugins/demo/enable"),
        ("GET", "/api/p/demo/hello"),
        ("POST", "/api/p/demo/items/a"),
        ("GET", "/plugins/demo/main.js"),
    ],
)
def test_paired_clients_never_reach_plugins(method: str, path: str) -> None:
    refused = remote.check_rights({"admin": True}, method, path, confirmed=True)
    assert refused is not None and refused[2] == "owner_only"
