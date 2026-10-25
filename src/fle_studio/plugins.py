"""Plugins: extra pages and API routes from installed Python packages.

A plugin is an installed package that names an entry point in the group `fle_studio.plugins`. The entry point's
name is the plugin's id (lower case letters, digits and `-`), its value a callable `register(ctx)`:

    [project.entry-points."fle_studio.plugins"]
    hello = "fle_studio_hello:register"

`register` gets a PluginContext and uses it to add a page in the sidebar (`ctx.page`), a folder of static files that
is served under `/plugins/<id>/` (`ctx.web`), and JSON routes under `/api/p/<id>/` (`ctx.route`). See docs/PLUGINS.md.

Safety model:
- Only installed packages are plugins; nothing is loaded from the network or from a folder.
- A plugin runs inside FLE Studio with its rights. So a plugin of someone else is off until it is turned on in
  Settings, one by one, and the list of plugins that are on lives in `plugins.json` in the data folder.
- The plugins that ship in FLE Studio's own package (Forge, Updates, Remote access, the Recorder, the model
  switcher: the entry points of the distribution `fle-studio` itself) are on by default and can be turned off one by
  one in Settings (`plugins.json` then lists them under "disabled"). The environment variable FLE_STUDIO_NO_BUNDLED=1
  leaves them out altogether: FLE Studio then runs with its core only.
- Plugins belong to this machine: a paired client of a host can neither see nor use the host's plugins (remote.py).
- A plugin's routes get the same guard as every /api/ call (host check, access key, page token for changes).
- A plugin that fails to load is reported in Settings and does not stop FLE Studio.
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

GROUP = "fle_studio.plugins"
API_VERSION = 1  # what a plugin can rely on; raised when PluginContext or the page kit change in a breaking way
OWN_PACKAGE = "fle-studio"  # the distribution whose plugins ship with FLE Studio
NO_BUNDLED = "FLE_STUDIO_NO_BUNDLED"  # set to 1: the plugins of the package itself are not looked for at all

ID = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_FILE = re.compile(r"^[\w][\w./-]*$")
_METHODS = ("GET", "POST", "PUT", "DELETE")


class PluginError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


@dataclass
class Request:
    """One call of a plugin route."""

    method: str
    path: str  # the part after /api/p/<id>, always starting with "/"
    query: dict[str, str]
    params: dict[str, str]  # the named groups of the route's pattern
    _read_json: Callable[[], dict[str, Any]]
    disconnected: Callable[[], bool] = lambda: False

    def json(self) -> dict[str, Any]:
        """The request body as a JSON object (at most 1 MB)."""
        return self._read_json()


Handler = Callable[[Request], tuple[int, Any]]


@dataclass
class Page:
    title: str
    icon: str = "grid"
    script: str = "main.js"
    path: str = ""  # a plugin of the package itself may have a path of its own, like /forge, instead of /p/<id>
    nav: bool = True  # an entry in the sidebar menu; False for a page that something else links to


@dataclass
class PluginContext:
    """What a plugin gets in `register(ctx)`."""

    id: str
    data_dir: Path  # <data folder>/plugins/<id>/, created on first use; the plugin's own files go here
    studio_version: str
    api_version: int = API_VERSION
    studio: Any = (
        None  # the running Studio (server.py): its config, engine, library ... for plugins of the package itself
    )
    bundled: bool = False  # ships with FLE Studio's own package
    _page: Page | None = None
    _web: Any = None  # a Traversable or a Path: the folder with the plugin's static files
    _routes: list[tuple[str, re.Pattern[str], Handler]] = field(default_factory=list)
    _raw: list[tuple[str, re.Pattern[str], Callable[..., None]]] = field(default_factory=list)
    _boot: str = ""
    _title: str = ""
    _summary: str = ""
    _hooks: dict[str, list[Callable[..., Any]]] = field(default_factory=dict)
    _stops: list[Callable[[], None]] = field(default_factory=list)

    def page(self, title: str, icon: str = "grid", script: str = "main.js", path: str = "", nav: bool = True) -> None:
        """Add an entry to the sidebar that opens /p/<id>. The page loads /plugins/<id>/<script> as an ES module and
        calls its `mount(main, kit)`; see docs/PLUGINS.md. `path` (only for the plugins of the package itself) puts
        the page at a path of its own, like /forge. `nav=False` leaves the entry out of the menu."""
        title = str(title).strip()[:40]
        if not title:
            raise ValueError("page: a title is needed")
        if not _FILE.match(script) or ".." in script or not script.endswith(".js"):
            raise ValueError("page: script must be a .js file in the plugin's web folder")
        if path and not (self.bundled and re.match(r"^/[a-z][a-z0-9-]{0,31}$", path)):
            raise ValueError("page: a path of its own is for the plugins of FLE Studio's package, like /forge")
        self._page = Page(title, str(icon or "grid"), script, path, bool(nav))

    def web(self, folder: Any) -> None:
        """The folder whose files are served under /plugins/<id>/.

        For example `importlib.resources.files(__package__) / "web"`."""
        self._web = folder

    def route(self, method: str, pattern: str, handler: Handler) -> None:
        """A JSON route under /api/p/<id>. `pattern` is a regular expression for the rest of the path, like
        r"^/items/(?P<item>[0-9a-f]{8})$"; the handler gets a Request and returns (status, data)."""
        method = method.upper()
        if method not in _METHODS:
            raise ValueError(f"route: method must be one of {', '.join(_METHODS)}")
        if not pattern.startswith("^/") or not pattern.endswith("$"):
            raise ValueError("route: the pattern must start with ^/ and end with $")
        self._routes.append((method, re.compile(pattern), handler))

    def api(self, method: str, pattern: str, fn: Callable[..., None]) -> None:
        """A route in FLE Studio's own /api space, for the plugins of the package itself (a plugin of someone else
        has `route`, under /api/p/<id>). `fn(handler, query, **groups)` gets the server's request handler and answers
        with `handler.send_json(...)`; it raises server.ApiError for a refusal. The pattern is an absolute path regex
        like r"^/api/forge/jobs/(?P<id>[0-9a-f]{12})$". The routes of a plugin that is off do not exist."""
        method = method.upper()
        if method not in _METHODS:
            raise ValueError(f"api: method must be one of {', '.join(_METHODS)}")
        if not pattern.startswith("^/api/") or not pattern.endswith("$"):
            raise ValueError("api: the pattern must start with ^/api/ and end with $")
        self._raw.append((method, re.compile(pattern), fn))

    def boot(self, script: str = "boot.js") -> None:
        """A script of the web folder that runs when the page starts, on every page (not only on the plugin's own):
        `export function init(kit)`. It adds what a plugin shows outside its page: a section in Settings, a
        notice in the sidebar. See docs/PLUGINS.md."""
        if not _FILE.match(script) or ".." in script or not script.endswith(".js"):
            raise ValueError("boot: script must be a .js file in the plugin's web folder")
        self._boot = script

    def describe(self, title: str, summary: str = "") -> None:
        """The name and a line about the plugin for the list in Settings (a plugin without a page needs them)."""
        self._title, self._summary = str(title).strip()[:40], str(summary).strip()[:200]

    def hook(self, name: str, fn: Callable[..., Any]) -> None:
        """Take part in something the core does, at a point it names (see HOOKS in this file). Hooks are for the
        plugins of the package itself."""
        if name not in HOOKS:
            raise ValueError(f"hook: unknown hook {name!r}; the core offers {', '.join(sorted(HOOKS))}")
        self._hooks.setdefault(name, []).append(fn)

    def on_stop(self, fn: Callable[[], None]) -> None:
        """Called when the plugin is turned off and when FLE Studio stops; end threads and close connections here."""
        self._stops.append(fn)

    def read_settings(self) -> dict[str, Any]:
        """The plugin's own settings (settings.json in its data folder), {} when there are none."""
        path = self.data_dir / "settings.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def write_settings(self, data: dict[str, Any]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        path = self.data_dir / "settings.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


@dataclass
class Found:
    """A plugin that is installed: from an entry point, or given by a test."""

    id: str
    load: Callable[[], Any]
    package: str = ""
    version: str = ""
    bundled: bool = False  # from FLE Studio's own package: on unless turned off


# What the core lets a plugin of the package itself take part in: name -> what the hook gets and gives.
HOOKS: dict[str, str] = {
    "proxy_record": "(path, body) -> recorder | None, when a request goes through /v1. A recorder has feed(chunk), "
    "done(status, body=None), wants_stream() and, once a stream is read, timings and usage.",
}


def discover() -> list[Found]:
    """Every installed plugin, sorted by id. Entry points with an invalid name are left out, and so are the plugins of
    FLE Studio's own package when FLE_STUDIO_NO_BUNDLED is set."""
    skip_own = os.environ.get(NO_BUNDLED, "").strip().lower() in ("1", "true", "yes", "on")
    out: dict[str, Found] = {}
    for ep in metadata.entry_points(group=GROUP):
        if not ID.match(ep.name) or ep.name in out:
            continue
        dist = getattr(ep, "dist", None)
        package = getattr(dist, "name", "") or ""
        own = package.lower().replace("_", "-") == OWN_PACKAGE
        if own and skip_own:
            continue
        out[ep.name] = Found(
            ep.name,
            ep.load,
            package=package,
            version=getattr(dist, "version", "") or "",
            bundled=own,
        )
    return [out[k] for k in sorted(out)]


class Plugins:
    """The plugins of one FLE Studio run."""

    def __init__(
        self,
        data_dir: Path,
        studio_version: str,
        find: Callable[[], Iterable[Found]] | None = None,
        studio: Any = None,
    ) -> None:
        self.data_dir = data_dir
        self.studio_version = studio_version
        self.studio = studio
        self._find = find or discover
        self._lock = threading.RLock()
        self.found: dict[str, Found] = {}
        self.loaded: dict[str, PluginContext] = {}
        self.errors: dict[str, str] = {}

    # -------------------------------------------------------------------------------------------- state

    @property
    def _list_file(self) -> Path:
        return self.data_dir / "plugins.json"

    def _read_list(self, key: str) -> list[str]:
        try:
            data = json.loads(self._list_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        ids = data.get(key) if isinstance(data, dict) else None
        return [i for i in ids if isinstance(i, str) and ID.match(i)] if isinstance(ids, list) else []

    def enabled(self) -> list[str]:
        """The plugins turned on by hand (those of other packages), and those of FLE Studio's own that were turned off
        and on again."""
        return self._read_list("enabled")

    def disabled(self) -> list[str]:
        """The plugins of FLE Studio's own package that were turned off."""
        return self._read_list("disabled")

    def is_on(self, pid: str) -> bool:
        found = self.found.get(pid)
        if found is not None and found.bundled:
            return pid not in self.disabled()
        return pid in self.enabled()

    def _save_lists(self, enabled: list[str], disabled: list[str]) -> None:
        tmp = self._list_file.with_suffix(".tmp")
        data: dict[str, Any] = {"enabled": sorted(set(enabled))}
        if disabled:
            data["disabled"] = sorted(set(disabled))
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._list_file)

    # -------------------------------------------------------------------------------------------- life

    def start(self) -> None:
        """Find the installed plugins and load those that are on."""
        with self._lock:
            try:
                self.found = {f.id: f for f in self._find()}
            except Exception as exc:  # a broken package must not stop FLE Studio
                self.found = {}
                self.errors["*"] = f"Looking for plugins failed: {exc}"
            for pid in self.found:
                if self.is_on(pid):
                    self._load(pid)

    def _load(self, pid: str) -> None:
        found = self.found[pid]
        ctx = PluginContext(
            pid, self.data_dir / "plugins" / pid, self.studio_version, studio=self.studio, bundled=found.bundled
        )
        try:
            register = found.load()
            if not callable(register):
                register = getattr(register, "register", None)
            if not callable(register):
                raise TypeError("the entry point is neither register(ctx) nor a module with register(ctx)")
            ctx.data_dir.mkdir(parents=True, exist_ok=True)
            register(ctx)
        except Exception as exc:
            traceback.print_exc()
            self.errors[pid] = f"{type(exc).__name__}: {exc}"
            self._stop(ctx)
            return
        self.errors.pop(pid, None)
        self.loaded[pid] = ctx

    def _stop(self, ctx: PluginContext) -> None:
        for fn in ctx._stops:
            try:
                fn()
            except Exception:
                traceback.print_exc()

    def set_enabled(self, pid: str, on: bool) -> None:
        with self._lock:
            if pid not in self.found:
                raise PluginError(404, f"No plugin named {pid} is installed")
            ids = [i for i in self.enabled() if i != pid] + ([pid] if on else [])
            off = [i for i in self.disabled() if i != pid] + ([] if on else [pid] if self.found[pid].bundled else [])
            self._save_lists(ids, off)
            if on and pid not in self.loaded:
                self._load(pid)
            if not on:
                ctx = self.loaded.pop(pid, None)
                self.errors.pop(pid, None)
                if ctx is not None:
                    self._stop(ctx)

    def stop(self) -> None:
        with self._lock:
            for ctx in self.loaded.values():
                self._stop(ctx)
            self.loaded.clear()

    # -------------------------------------------------------------------------------------------- views

    def view(self) -> dict[str, Any]:
        """For Settings and the sidebar: every installed plugin, whether it is on, and its page."""
        items = []
        for pid, f in sorted(self.found.items()):
            ctx = self.loaded.get(pid)
            page = ctx._page if ctx else None
            items.append(
                {
                    "id": pid,
                    "package": f.package,
                    "version": f.version,
                    "bundled": f.bundled,
                    "enabled": self.is_on(pid),
                    "loaded": ctx is not None,
                    "error": self.errors.get(pid, ""),
                    "page": {
                        "title": page.title,
                        "icon": page.icon,
                        "script": page.script,
                        "path": page.path,
                        "nav": page.nav,
                    }
                    if page
                    else None,
                    "boot": ctx._boot if ctx else "",
                    "title": ctx._title if ctx else "",
                    "summary": ctx._summary if ctx else "",
                }
            )
        return {"api_version": API_VERSION, "plugins": items, "error": self.errors.get("*", "")}

    # -------------------------------------------------------------------------------------------- serving

    def _ctx(self, pid: str) -> PluginContext:
        ctx = self.loaded.get(pid)
        if ctx is None:
            raise PluginError(404, f"The plugin {pid} is not on")
        return ctx

    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, str],
        read_json: Callable[[], dict[str, Any]],
        disconnected: Callable[[], bool] = lambda: False,
    ) -> tuple[int, Any]:
        """Answer /api/p/<id>/<rest>."""
        m = re.match(r"^/api/p/([a-z][a-z0-9-]{0,31})(/.*)?$", path)
        if not m:
            raise PluginError(404, "Not found")
        ctx = self._ctx(m.group(1))
        rest = m.group(2) or "/"
        allowed = False
        for meth, rx, fn in ctx._routes:
            hit = rx.match(rest)
            if not hit:
                continue
            allowed = True
            if meth == method:
                req = Request(method, rest, query, hit.groupdict(), read_json, disconnected)
                status, data = fn(req)
                return int(status), data
        if allowed:
            raise PluginError(405, "Method not allowed")
        raise PluginError(404, "Not found")

    def handle_raw(self, handler: Any, method: str, path: str, query: dict[str, str]) -> bool:
        """Answer a request with a route that a plugin of the package itself put into /api (ctx.api). False when
        none of the plugins that are on has such a route."""
        for ctx in list(self.loaded.values()):
            for meth, rx, fn in ctx._raw:
                hit = rx.match(path)
                if hit and meth == method:
                    fn(handler, query, **hit.groupdict())
                    return True
        return False

    def hooks(self, name: str) -> list[Callable[..., Any]]:
        """The functions the plugins that are on have put at this hook, in the order of the plugin ids."""
        return [fn for pid in sorted(self.loaded) for fn in self.loaded[pid]._hooks.get(name, [])]

    def static(self, path: str) -> tuple[bytes, str]:
        """A file under /plugins/<id>/; returns (bytes, file name)."""
        m = re.match(r"^/plugins/([a-z][a-z0-9-]{0,31})/(.+)$", path)
        if not m:
            raise PluginError(404, "Not found")
        ctx = self._ctx(m.group(1))
        name = m.group(2)
        if ctx._web is None or not _FILE.match(name) or ".." in name.split("/"):
            raise PluginError(404, "Not found")
        node = ctx._web
        for part in name.split("/"):
            node = node / part
        try:
            return node.read_bytes(), name
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError):
            raise PluginError(404, "Not found") from None


# ---------------------------------------------------------------------------------------------- HTTP

ROUTES = [
    ("GET", r"^/api/plugins$", "plugins"),
    ("POST", r"^/api/plugins/(?P<pid>[a-z][a-z0-9-]{0,31})/(?P<action>enable|disable)$", "plugins_toggle"),
]
_ApiError: type = Exception


def api_plugins(self: Any, query: dict[str, str]) -> None:
    self.send_json(200, self.studio.plugins.view())


def api_plugins_toggle(self: Any, query: dict[str, str], pid: str, action: str) -> None:
    try:
        self.studio.plugins.set_enabled(pid, action == "enable")
    except PluginError as exc:
        raise _ApiError(exc.status, exc.message) from None
    self.send_json(200, self.studio.plugins.view())


def dispatch(handler: Any, path: str, query: dict[str, str]) -> None:
    """Answer a request under /api/p/ for the server's _dispatch."""
    try:
        status, data = handler.studio.plugins.handle(handler.command, path, query, handler.read_json)
    except PluginError as exc:
        raise _ApiError(exc.status, exc.message) from None
    handler.send_json(status, data)


def install(handler: type, routes: list[Any], api_error: type) -> None:
    """Add the handlers to the server's Handler class and the routes to its route table."""
    global _ApiError
    _ApiError = api_error
    for method, pattern, name in ROUTES:
        routes.append((method, re.compile(pattern), name))
        setattr(handler, f"api_{name}", globals()[f"api_{name}"])
