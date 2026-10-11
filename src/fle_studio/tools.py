"""Tools the model may call in a chat: a few built-in ones and the tools of MCP servers.

Each tool has a name, a description and a JSON schema for its arguments, which go to llama-server in the
OpenAI-style `tools` field of the chat request. The page runs the loop (model → tool → model); this module runs a
single tool call and says whether it needs the user's approval first.

Built-in tools are careful by default: the calculator and the clock work on nothing but their arguments; the file
tools read inside one folder the user picked and write nothing; web search is off until it is turned on, and asks
before each search. A tool of an MCP server may do anything its server does, so each call asks first, unless the
user allowed that server in the settings.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from . import net
from .config import Config
from .mcp import McpError, McpManager, parse_servers, result_text
from .oauth import OAuthError, OAuthManager
from .paths import IS_WIN

MAX_FILE_CHARS = 100_000
MAX_LIST_ENTRIES = 200


class ToolError(Exception):
    """A tool call failed in a way the model should read ("file not found"); not a bug of FLE Studio."""


class NeedsApproval(Exception):
    """The call needs the user's approval and the request did not carry it."""


@dataclass
class Tool:
    name: str  # as the model sees it
    description: str
    parameters: dict[str, Any]
    group: str  # calculator | clock | files | web_search | mcp
    source: str = "builtin"  # builtin | mcp
    server: str = ""
    remote_name: str = ""  # the tool's own name at its MCP server
    confirm: bool = False
    ui: str = ""  # the ui:// resource of an MCP App that shows this tool's result
    run: Callable[[dict[str, Any]], str] | None = field(default=None, repr=False)

    def openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "source": self.source,
            "server": self.server,
            "confirm": self.confirm,
            "ui": self.ui,
            "schema": self.openai(),
        }


# ------------------------------------------------------------------------------------------------ calculator

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
    "radians": math.radians,
    "degrees": math.degrees,
    "factorial": math.factorial,
    "gcd": math.gcd,
}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
MAX_EXPRESSION = 500


def _power(base: Any, exponent: Any) -> Any:
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        if abs(base).bit_length() * exponent > 20_000:
            raise ToolError("That power is too large to compute")
    return base**exponent


def _evaluate(node: ast.AST, depth: int = 0) -> Any:
    if depth > 40:
        raise ToolError("The expression is nested too deeply")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, depth + 1)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"Unknown name '{node.id}'. Known constants: pi, e, tau")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate(node.operand, depth + 1)
        return -value if isinstance(node.op, ast.USub) else +value
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left, depth + 1), _evaluate(node.right, depth + 1)
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        op = _BINARY.get(type(node.op))
        if op is None:
            raise ToolError("Only + - * / // % ** and parentheses are allowed")
        return op(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func = _FUNCTIONS.get(node.func.id)
        if func is None:
            raise ToolError(f"Unknown function '{node.func.id}'. Known: {', '.join(sorted(_FUNCTIONS))}")
        args = [_evaluate(a, depth + 1) for a in node.args]
        if node.func.id == "factorial" and args and isinstance(args[0], int) and args[0] > 1000:
            raise ToolError("factorial is limited to 1000")
        return func(*args)
    raise ToolError("Only numbers, + - * / // % ** , parentheses, constants and common functions are allowed")


def calculator(arguments: dict[str, Any]) -> str:
    text = str(arguments.get("expression") or "").strip()
    if not text:
        raise ToolError('Give the "expression" to calculate, for example "(3 + 4) * 2".')
    if len(text) > MAX_EXPRESSION:
        raise ToolError("The expression is too long")
    text = text.translate({ord("×"): "*", ord("÷"): "/", ord("−"): "-"}).replace("^", "**")  # how people write it
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ToolError(f"Cannot read the expression '{text}'") from None
    try:
        value = _evaluate(tree)
    except ZeroDivisionError:
        raise ToolError("Division by zero") from None
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"Cannot calculate that: {exc}") from None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ToolError("The result is not a finite number")
        return format(value, ".12g")
    return str(value)


# ------------------------------------------------------------------------------------------------ clock


def clock(arguments: dict[str, Any]) -> str:
    name = str(arguments.get("timezone") or "").strip()
    if not name:
        now = datetime.now().astimezone()
    elif name.upper() in ("UTC", "Z", "GMT"):
        now = datetime.now(timezone.utc)
    else:
        try:
            from zoneinfo import ZoneInfo  # noqa: PLC0415

            now = datetime.now(ZoneInfo(name))
        except Exception:  # unknown name, or Windows without the tzdata package
            raise ToolError(
                f"Unknown time zone '{name}'. Use a name like Europe/Berlin or UTC; "
                "leave it out for this computer's time."
            ) from None
    where = name or "this computer's local time"
    return f"{now.isoformat(timespec='seconds')} ({now.strftime('%A')}, {now.tzname()}, {where})"


# ------------------------------------------------------------------------------------------------ files


def _inside(root: Path, rel: str) -> Path:
    """`rel` resolved inside `root`, or ToolError. Symbolic links that lead out are refused too."""
    candidate = Path(rel)
    if "\x00" in rel or (IS_WIN and ":" in rel and not candidate.is_absolute()):  # no NTFS alternate data streams
        raise ToolError("That path is not allowed")
    full = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if full != root and root not in full.parents:
        raise ToolError("That path is outside the folder the user shared with the model")
    return full


def _files_root(folder: str) -> Path:
    if not folder.strip():
        raise ToolError("No folder is shared with the model")
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ToolError("The shared folder does not exist")
    return root.resolve()


def list_files(folder: str, arguments: dict[str, Any]) -> str:
    root = _files_root(folder)
    target = _inside(root, str(arguments.get("path") or "."))
    if not target.is_dir():
        raise ToolError("That is not a folder")
    rows = []
    entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for p in entries[:MAX_LIST_ENTRIES]:
        try:
            rows.append(f"{p.name}/" if p.is_dir() else f"{p.name}  ({p.stat().st_size} bytes)")
        except OSError:
            rows.append(p.name)
    more = f"\n… and {len(entries) - MAX_LIST_ENTRIES} more" if len(entries) > MAX_LIST_ENTRIES else ""
    shown = target.relative_to(root).as_posix() if target != root else "."
    return f"{shown}:\n" + ("\n".join(rows) if rows else "(empty)") + more


def read_file(folder: str, arguments: dict[str, Any]) -> str:
    root = _files_root(folder)
    path = str(arguments.get("path") or "").strip()
    if not path:
        raise ToolError('Give the "path" of the file, relative to the shared folder.')
    target = _inside(root, path)
    if not target.is_file():
        raise ToolError("There is no such file in the shared folder")
    try:
        offset = min(10_000_000, max(0, int(arguments.get("offset") or 0)))
        limit = max(1, min(MAX_FILE_CHARS, int(arguments.get("max_chars") or 20_000)))
    except (TypeError, ValueError):
        raise ToolError('"offset" and "max_chars" must be whole numbers') from None
    try:
        with target.open("rb") as fh:
            raw = fh.read(offset * 4 + limit * 4 + 8)  # enough bytes for any UTF-8 text of that many characters
    except OSError as exc:
        raise ToolError(f"Cannot read the file: {exc.strerror or exc}") from None
    if b"\x00" in raw[:8000]:
        raise ToolError("That is a binary file")
    text = raw.decode("utf-8", "replace")
    size = target.stat().st_size
    chunk = text[offset : offset + limit]
    more = len(text) > offset + limit or size > len(raw)
    note = f"\n[more follows: ask again with offset={offset + len(chunk)}]" if more else ""
    return chunk + note


# ------------------------------------------------------------------------------------------------ web search


class _DuckParser(HTMLParser):
    """Results of DuckDuckGo's plain HTML page: links with class result__a, snippets with class result__snippet."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._mode = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        a = {k: v or "" for k, v in attrs}
        classes = a.get("class", "").split()
        if "result__a" in classes:
            href = a.get("href", "")
            target = parse_qs(urlparse(href).query).get("uddg", [href])[0]
            self.results.append({"title": "", "url": target, "snippet": ""})
            self._mode = "title"
        elif "result__snippet" in classes and self.results:
            self._mode = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._mode = ""

    def handle_data(self, data: str) -> None:
        if self._mode and self.results:
            self.results[-1][self._mode] += data


def parse_duckduckgo(html: str) -> list[dict[str, str]]:
    parser = _DuckParser()
    parser.feed(html)
    return [{k: " ".join(v.split()) for k, v in r.items()} for r in parser.results if r["url"]]


def parse_searxng(text: str) -> list[dict[str, str]]:
    try:
        doc = json.loads(text)
    except ValueError:
        raise ToolError("The search server did not answer with JSON (is format=json allowed there?)") from None
    out = []
    for item in doc.get("results") or [] if isinstance(doc, dict) else []:
        if isinstance(item, dict) and item.get("url"):
            out.append(
                {
                    "title": " ".join(str(item.get("title") or "").split()),
                    "url": str(item["url"]),
                    "snippet": " ".join(str(item.get("content") or "").split()),
                }
            )
    return out


def web_search(base_url: str, arguments: dict[str, Any]) -> str:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ToolError('Give the "query" to search for.')
    try:
        count = max(1, min(10, int(arguments.get("max_results") or 5)))
    except (TypeError, ValueError):
        count = 5
    custom = base_url.strip()
    if custom:
        if urlparse(custom).scheme not in ("http", "https"):
            raise ToolError("The search server address must start with http:// or https://")
        url = f"{custom}{'&' if '?' in custom else '?'}q={quote(query)}&format=json"
    else:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    try:
        with net.open_url(url, timeout=15, headers={"Accept": "text/html,application/json"}) as resp:
            body = resp.read(1_500_000).decode("utf-8", "replace")
    except OSError as exc:
        raise ToolError(f"The search failed: {getattr(exc, 'reason', exc)}") from None
    results = (parse_searxng(body) if custom else parse_duckduckgo(body))[:count]
    if not results:
        return "No results."
    return "\n\n".join(
        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}".rstrip() for i, r in enumerate(results, 1)
    )


# ------------------------------------------------------------------------------------------------ the set


def _simplify_schema(schema: Any) -> dict[str, Any]:
    """An MCP input schema as llama-server likes it: an object schema without the `$schema` marker."""
    if not isinstance(schema, dict) or schema.get("type") not in (None, "object"):
        return {"type": "object", "properties": {}}
    out = {k: v for k, v in schema.items() if k not in ("$schema", "$id")}
    out["type"] = "object"
    out.setdefault("properties", {})
    return out


def _wire_name(server: str, tool: str, taken: set[str]) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}")
    base = clean[:64]
    name, n = base, 2
    while name in taken:
        suffix = f"_{n}"
        name, n = base[: 64 - len(suffix)] + suffix, n + 1
    return name


def tool_visibility(t: dict[str, Any]) -> list[str]:
    """Who may call a tool (MCP Apps, `_meta.ui.visibility`): the model, the app, or both (the default)."""
    meta = t.get("_meta") if isinstance(t.get("_meta"), dict) else {}
    ui = meta.get("ui") if isinstance(meta.get("ui"), dict) else {}
    vis = ui.get("visibility")
    if isinstance(vis, list) and vis:
        return [str(v) for v in vis if v in ("model", "app")]
    return ["model", "app"]


def tool_ui_uri(t: dict[str, Any]) -> str:
    """The ui:// resource that shows a tool's result: `_meta.ui.resourceUri`, the older flat `ui/resourceUri`, or
    the ChatGPT Apps SDK's `openai/outputTemplate`. Anything that is not a ui:// address is ignored."""
    meta = t.get("_meta") if isinstance(t.get("_meta"), dict) else {}
    ui = meta.get("ui") if isinstance(meta.get("ui"), dict) else {}
    for value in (ui.get("resourceUri"), meta.get("ui/resourceUri"), meta.get("openai/outputTemplate")):
        if isinstance(value, str) and value.startswith("ui://"):
            return value
    return ""


def _app_result(raw: dict[str, Any]) -> dict[str, Any]:
    """A CallToolResult for an app, kept to a size the page can handle."""
    text = json.dumps(raw, ensure_ascii=False, default=str)
    if len(text) <= 2_000_000:
        return raw
    return {"content": [{"type": "text", "text": "[the result was too large to hand to the app]"}], "isError": True}


class ToolBox:
    """The tools that are on right now, read from the settings every time."""

    def __init__(self, config: Config, data_dir: Path | None = None) -> None:
        self.config = config
        self.oauth = OAuthManager(data_dir) if data_dir is not None else None
        self.mcp = McpManager(self.oauth)
        self._synced = ""

    # ---- settings

    def settings(self) -> dict[str, Any]:
        return dict(self.config.get("tools"))

    def trusted(self) -> set[str]:
        raw = self.settings().get("mcp_trusted")
        return {str(x) for x in raw} if isinstance(raw, list) else set()

    def files_dir_ok(self) -> bool:
        folder = str(self.settings().get("files_dir") or "").strip()
        return bool(folder) and Path(folder).expanduser().is_dir()

    def sync_mcp(self) -> dict[str, str]:
        """Make the connected servers match the settings. Returns the problems found in the JSON text."""
        s = self.settings()
        servers, problems = parse_servers(str(s.get("mcp_json") or ""))
        if not (s.get("enabled") and s.get("mcp_enabled")):
            servers = {}
        self.mcp.sync(servers, float(s.get("call_timeout") or 60))
        return problems

    # ---- listing

    def tools(self) -> list[Tool]:
        s = self.settings()
        out: list[Tool] = []
        if not s.get("enabled"):
            return out
        if s.get("calculator"):
            out.append(
                Tool(
                    "calculator",
                    "Calculate an arithmetic expression exactly. Use it for any maths instead "
                    "of working it out yourself.",
                    {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "For example (12.5 + 7) * 3 / 4 or sqrt(2)**2",
                            }
                        },
                        "required": ["expression"],
                    },
                    "calculator",
                    run=calculator,
                )
            )
        if s.get("datetime"):
            out.append(
                Tool(
                    "get_current_time",
                    "Get the current date and time.",
                    {
                        "type": "object",
                        "properties": {
                            "timezone": {
                                "type": "string",
                                "description": "Optional IANA time zone such as Europe/Berlin "
                                "or UTC. Omit it for this computer's local time.",
                            }
                        },
                    },
                    "clock",
                    run=clock,
                )
            )
        if s.get("files") and self.files_dir_ok():
            folder = str(s["files_dir"])
            out.append(
                Tool(
                    "list_files",
                    "List the files and folders in a folder that the user shared with you.",
                    {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Folder relative to the shared folder; "
                                "omit it for the shared folder itself.",
                            }
                        },
                    },
                    "files",
                    run=lambda a, f=folder: list_files(f, a),
                )
            )
            out.append(
                Tool(
                    "read_file",
                    "Read a text file from the folder that the user shared with you.",
                    {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path relative to the shared folder."},
                            "offset": {"type": "integer", "description": "First character to return (default 0)."},
                            "max_chars": {
                                "type": "integer",
                                "description": "How many characters to return (default 20000).",
                            },
                        },
                        "required": ["path"],
                    },
                    "files",
                    run=lambda a, f=folder: read_file(f, a),
                )
            )
        if s.get("web_search"):
            url = str(s.get("web_search_url") or "")
            out.append(
                Tool(
                    "web_search",
                    "Search the web and return titles, links and short snippets.",
                    {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "description": "1 to 10, default 5"},
                        },
                        "required": ["query"],
                    },
                    "web_search",
                    confirm=bool(s.get("web_search_confirm")),
                    run=lambda a, u=url: web_search(u, a),
                )
            )
        if s.get("mcp_enabled"):
            self.sync_mcp()
            taken = {t.name for t in out}
            trusted = self.trusted()
            for server in self.mcp.servers():
                if server.status != "ready":
                    continue
                for t in server.tools:
                    if "model" not in tool_visibility(t):  # a tool only the app may call
                        continue
                    name = _wire_name(server.name, str(t["name"]), taken)
                    taken.add(name)
                    desc = str(t.get("description") or t.get("title") or t["name"])
                    out.append(
                        Tool(
                            name,
                            f"[{server.name}] {desc}"[:1500],
                            _simplify_schema(t.get("inputSchema")),
                            "mcp",
                            source="mcp",
                            server=server.name,
                            remote_name=str(t["name"]),
                            confirm=server.name not in trusted,
                            ui=tool_ui_uri(t),
                        )
                    )
        return out

    def status(self) -> dict[str, Any]:
        """What the page needs: the tools, the MCP servers and why something is missing."""
        s = self.settings()
        problems = self.sync_mcp()
        if not (s.get("enabled") and s.get("mcp_enabled")):
            problems = {}  # the servers are off, so a mistake in their text does not matter yet
        trusted = self.trusted()
        servers = [
            {
                "name": sv.name,
                "status": sv.status,
                "error": sv.error,
                "tools": len(sv.tools),
                "transport": sv.spec["transport"],
                "trusted": sv.name in trusted,
                "info": (sv.info.get("name") or "") if isinstance(sv.info, dict) else "",
                "auth": self.auth_state(sv),
                "legacy_sse": sv.fallback,
            }
            for sv in self.mcp.servers()
        ]
        for name, why in problems.items():
            servers.append(
                {
                    "name": name or "(settings)",
                    "status": "error",
                    "error": why,
                    "tools": 0,
                    "transport": "",
                    "trusted": False,
                    "info": "",
                    "auth": {"remote": False},
                    "legacy_sse": False,
                }
            )
        return {
            "enabled": bool(s.get("enabled")),
            "max_rounds": int(s.get("max_rounds") or 8),
            "tools": [t.public() for t in self.tools()],
            "servers": servers,
            "connecting": any(sv.status in ("idle", "connecting") for sv in self.mcp.servers()),
            "files_dir_ok": self.files_dir_ok(),
        }

    # ---- calling

    def call(self, name: str, arguments: Any, approved: bool = False) -> dict[str, Any]:
        """Run one tool. Returns {"content", "is_error", "ms"}. Raises NeedsApproval or KeyError (unknown tool)."""
        tool = next((t for t in self.tools() if t.name == name), None)
        if tool is None:
            raise KeyError(name)
        if tool.confirm and not approved:
            raise NeedsApproval(name)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except ValueError:
                return {"content": "The arguments are not valid JSON.", "is_error": True, "ms": 0}
        if not isinstance(arguments, dict):
            return {"content": "The arguments must be a JSON object.", "is_error": True, "ms": 0}
        started = time.monotonic()
        app_result: dict[str, Any] | None = None
        try:
            if tool.source == "mcp":
                server = self.mcp.get(tool.server)
                if server is None:
                    raise ToolError(f"The server '{tool.server}' is not connected")
                try:
                    raw = server.call_raw(tool.remote_name, arguments)
                except McpError as exc:
                    raise ToolError(str(exc)) from None
                content, is_error = result_text(raw)
                if tool.ui:  # the app gets the whole result (structuredContent, _meta), not the model's text
                    app_result = _app_result(raw)
            else:
                assert tool.run is not None
                content, is_error = tool.run(arguments), False
        except ToolError as exc:
            content, is_error = str(exc), True
        except Exception as exc:  # a bug in a tool must not take the request down; the model gets the message
            content, is_error = f"The tool failed: {type(exc).__name__}: {exc}", True
        out = {"content": content, "is_error": is_error, "ms": int((time.monotonic() - started) * 1000)}
        if app_result is not None:
            out["app"] = {"server": tool.server, "uri": tool.ui, "tool": tool.remote_name, "result": app_result}
        return out

    # ---- MCP: sign-in and apps

    def auth_state(self, server: Any) -> dict[str, Any]:
        """What the settings page shows about a server's sign-in. Never contains a token."""
        if server.spec["transport"] == "stdio":
            return {"remote": False}
        info: dict[str, Any] = {"remote": True, "required": server.status == "auth_required"}
        if self.oauth is not None:
            info.update(self.oauth.status(server.name, server.spec["url"]))
        info["client_id_set"] = bool((server.spec.get("oauth") or {}).get("client_id"))
        return info

    def server_for_signin(self, name: str) -> Any:
        server = self.mcp.get(name)
        if server is None or server.spec["transport"] == "stdio":
            raise KeyError(name)
        if self.oauth is None:
            raise OAuthError("Signing in is not available here.")
        return server

    def begin_sign_in(self, name: str, redirect_uri: str) -> dict[str, Any]:
        server = self.server_for_signin(name)
        assert self.oauth is not None
        preset = server.spec.get("oauth") or {}
        return self.oauth.begin(name, server.spec["url"], redirect_uri, challenge=server.challenge, preset=preset)

    def finish_sign_in(self, state: str, code: str, error: str, description: str) -> str:
        assert self.oauth is not None
        name = self.oauth.complete(state, code, error, description)
        self.mcp.reconnect(name)
        return name

    def sign_out(self, name: str) -> bool:
        server = self.server_for_signin(name)
        assert self.oauth is not None
        removed = self.oauth.sign_out(name, server.spec["url"])
        self.mcp.reconnect(name)
        return removed

    def app_tool(self, server_name: str, remote_name: str) -> dict[str, Any]:
        """The server's tool description when an app may call it (visibility holds "app")."""
        server = self.mcp.get(server_name)
        if server is None or server.status != "ready":
            raise KeyError(server_name)
        for t in server.tools:
            if t["name"] == remote_name and "app" in tool_visibility(t):
                return t
        raise KeyError(remote_name)

    def app_call(self, server_name: str, remote_name: str, arguments: Any, approved: bool = False) -> dict[str, Any]:
        """A tools/call made by an MCP App: only to its own server, only tools meant for apps, and with the same
        confirmation as the model's calls (unless the server is trusted)."""
        self.app_tool(server_name, remote_name)
        if server_name not in self.trusted() and not approved:
            raise NeedsApproval(remote_name)
        if not isinstance(arguments, dict):
            arguments = {}
        server = self.mcp.get(server_name)
        assert server is not None
        try:
            return _app_result(server.call_raw(remote_name, arguments))
        except McpError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}

    def close(self) -> None:
        self.mcp.close_all()


# ------------------------------------------------------------------------------------------------ chat requests


FORMAT_FIELDS = ("response_format", "json_schema", "grammar")


def forced_format(body: dict[str, Any]) -> list[str]:
    """The fields of a chat request that force the shape of the answer (JSON mode, JSON schema, grammar)."""
    found = []
    for key in FORMAT_FIELDS:
        value = body.get(key)
        if key == "response_format" and isinstance(value, dict) and value.get("type") in (None, "text"):
            continue
        if value not in (None, "", {}, []):
            found.append(key)
    return found


def check_tools_and_format(body: dict[str, Any]) -> None:
    """Raise ValueError when one request asks for tools and for a forced answer format.

    llama-server then builds a grammar for the answer format and the model does not call the tools; it answers
    with substitute JSON and nothing says why. Ask for one or the other, in two requests.
    """
    tools = body.get("tools")
    if not tools or body.get("tool_choice") == "none":
        return
    clash = forced_format(body)
    if clash:
        raise ValueError(
            f"A chat request must not carry both tools and {', '.join(clash)}: the model would not call "
            "the tools and would answer with substitute JSON. Send them in separate requests."
        )
