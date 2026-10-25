# Plugins: your own pages in FLE Studio

A plugin adds a page to FLE Studio without changing FLE Studio itself: an entry in the sidebar, the page's script and
styles, and the JSON routes that page talks to. FLE Studio stays as it is; what is special to you lives in your own
package.

```
sidebar entry ──> /p/<id> ──> /plugins/<id>/main.js  (your page, an ES module)
                                     │ kit.api.get("/…")
                                     ▼
                              /api/p/<id>/…          (your routes, Python, inside FLE Studio)
```

## What a plugin is

An installed Python package that names an entry point in the group `fle_studio.plugins`. The entry point's name is the
plugin's id (lower case letters, digits and `-`, up to 32 characters), its value a function `register(ctx)`:

```toml
# pyproject.toml of your package
[project.entry-points."fle_studio.plugins"]
hello = "fle_studio_hello:register"
```

```python
# fle_studio_hello/__init__.py
from importlib import resources


def register(ctx):
    ctx.page("Hello", icon="spark")                      # the sidebar entry; opens /p/hello
    ctx.web(resources.files(__package__) / "web")         # served under /plugins/hello/
    ctx.route("GET", r"^/greeting$", lambda req: (200, {"text": f"Hello, {req.query.get('name', 'you')}"}))
```

Install it into the same Python environment as FLE Studio, for example with uv:

```sh
uv tool install --with ./fle-studio-hello fle-studio      # or: uv tool install --with fle-studio-hello fle-studio
```

Restart FLE Studio, then turn the plugin on in **Settings → Plugins**.

## The context `ctx`

| | |
| --- | --- |
| `ctx.id` | the plugin's id |
| `ctx.data_dir` | `<data folder>/plugins/<id>/`, the plugin's own folder |
| `ctx.studio_version`, `ctx.api_version` | FLE Studio's version and the plugin API's version (now 1) |
| `ctx.page(title, icon="grid", script="main.js")` | the sidebar entry; `icon` is one of FLE Studio's icon names |
| `ctx.web(folder)` | the folder served under `/plugins/<id>/` (a `Path` or an `importlib.resources` folder) |
| `ctx.route(method, pattern, handler)` | a JSON route under `/api/p/<id>`; `pattern` is a regular expression for the rest of the path, from `^/` to `$`; named groups land in `req.params` |
| `ctx.on_stop(fn)` | called when the plugin is turned off and when FLE Studio stops |
| `ctx.read_settings()`, `ctx.write_settings(data)` | the plugin's own `settings.json` in its folder |

A handler gets a `Request` (`method`, `path`, `query`, `params`, `json()`) and returns `(status, data)`; FLE Studio
sends `data` as JSON. Raise nothing for expected problems: answer with a status and a message instead. Each request
runs in its own thread, so a route may wait (a long poll) without holding up the rest of FLE Studio.

## The page

`/p/<id>` loads the script that `ctx.page` named and calls its `mount(main, kit)`. `main` is the page's area; `mount`
may return `{ unmount() }`, which runs when the person leaves the page.

```js
// fle_studio_hello/web/main.js
export function mount(main, kit) {
  const out = kit.h("p", null, "…");
  main.replaceChildren(kit.topbar("Hello"), kit.h("div", { class: "page" }, kit.h("div", { class: "page-inner" }, out)));
  kit.api.get("/greeting?name=Max").then((r) => { out.textContent = r.text; }, (e) => kit.toast(e.message, "bad"));
  return { unmount() {} };
}
```

The kit (`KIT_VERSION` 1) holds what a page should use, and stays stable within one version:

| | |
| --- | --- |
| `kit.api.get/post/put/del(path, body)` | calls to the plugin's own routes, with the page token; always on this machine |
| `kit.asset(name)` | the address of a file in the plugin's web folder |
| `kit.h`, `kit.clear`, `kit.icon`, `kit.fmt`, `kit.toast`, `kit.confirmDialog` | FLE Studio's own building blocks |
| `kit.topbar(title, ...right)`, `kit.navigate(path)`, `kit.on(topic, fn)`, `kit.theme()` | the page frame, moving to another page, events (`"state"`, `"engine"`, `"live"`), the theme |

FLE Studio's style sheet applies to plugin pages, so classes like `page`, `page-inner`, `card`, `btn`, `pill` and `row`
look as they do everywhere else. A plugin may add its own style sheet with a `<link>` to `kit.asset("style.css")`.

## Plugins that ship with FLE Studio

Parts of FLE Studio that not everybody needs are plugins of FLE Studio's own package, so the core stays small:

| id | what it is |
| --- | --- |
| `forge` | the Forge page and its `/api/forge` routes: build your own FLE quant (docs/FORGE.md) |
| `updater` | the daily look for a newer FLE Studio or engine, the notice and the one-click update |
| `switch` | the model switcher next to an attached service, in the chat's model menu |

They are separate Python packages (`fle_studio_<id>`) in the same wheel, registered through the entry point group
`fle_studio.plugins` of the distribution `fle-studio` like any plugin. They are **on by default**; Settings → Plugins
turns each one off (the page reloads), and turning it on again brings it back. `plugins.json` then holds a
`"disabled"` list next to `"enabled"`. With the environment variable `FLE_STUDIO_NO_BUNDLED=1` they are not even
looked for: FLE Studio runs with its core alone, and nothing in the core breaks for their absence (the CI runs the core
tests like this).

For these plugins the context offers a little more than for the plugins of others. They may change in any release:

| | |
| --- | --- |
| `ctx.studio` | the running `Studio` (settings, engine, library …) |
| `ctx.api(method, pattern, fn)` | a route in FLE Studio's own `/api` space; `fn(handler, query, **groups)` answers with `handler.send_json(...)` |
| `ctx.boot("boot.js")` | a script that runs on every page when it starts: `export function init(kit)`; it adds Settings sections (`kit.settingsSection`), model menu entries (`modelMenuProviders` in app.js), sidebar notices |
| `ctx.page(title, icon, script, path="/forge")` | a page at a path of its own |
| `ctx.describe(title, summary)` | the name and a line for the list in Settings |
| `ctx.hook(name, fn)` | take part in something the core does at a named point |

Their scripts may import FLE Studio's own modules (`/js/ui.js`, `/js/api.js`, `/js/app.js`); the scripts of other
plugins use the kit.

## Safety

- **Only installed packages.** FLE Studio loads nothing from the network or from a folder.
- **Off until turned on.** A plugin runs inside FLE Studio with its rights. Each plugin of someone else is turned on
  by hand in Settings; the list of plugins that are on lives in `plugins.json` in the data folder. (The plugins of
  FLE Studio's own package are on by default, as above: they are part of the same download.)
- **Same guard as everything else.** Plugin routes need the access key in LAN mode, and every change (POST, PUT,
  DELETE) needs the page token. Pages get the same Content-Security-Policy as FLE Studio's own: scripts and connections
  only from FLE Studio itself.
- **This machine only.** A paired client of a host neither sees nor uses the host's plugins, and the page calls its
  plugins on its own Studio even while another host is chosen.
- **A broken plugin** is shown in Settings with its error; FLE Studio goes on without it.
- Secrets a plugin needs (a token for a service it talks to) belong in its own folder with tight rights, never in a
  page, a log or an address.
