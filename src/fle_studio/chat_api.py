"""The HTTP side of the chat features: attachments, tools, search, export and import.

Kept out of server.py on purpose: `install` adds these handlers and routes to the server's `Handler` and `ROUTES`,
so server.py only has to call it once.
"""

from __future__ import annotations

import importlib.util
import json
import re
from typing import Any
from urllib.parse import quote

from . import __version__, attachments, chat_tree, tools

ROUTES: list[tuple[str, str, str]] = [
    ("GET", r"^/api/chat/capabilities$", "chat_caps"),
    ("POST", r"^/api/attachments$", "attachment_upload"),
    ("GET", r"^/api/attachments/(?P<name>[0-9a-f]{64}\.(?:png|jpg|gif|webp))$", "attachment_get"),
    ("GET", r"^/api/chats/search$", "chat_search"),
    ("GET", r"^/api/chats/(?P<id>[0-9a-f]{16})/export$", "chat_export"),
    ("POST", r"^/api/chats/import$", "chat_import"),
    ("GET", r"^/api/tools$", "tools"),
    ("POST", r"^/api/tools/call$", "tools_call"),
    ("POST", r"^/api/tools/reload$", "tools_reload"),
]

_ApiError: Any = None  # server.ApiError, handed in by install()


def _fail(status: int, message: str) -> Exception:
    return _ApiError(status, message)


def toolbox(studio: Any) -> tools.ToolBox:
    box = studio.__dict__.get("toolbox")
    if box is None:
        box = studio.__dict__["toolbox"] = tools.ToolBox(studio.config, studio.data_dir)
    return box  # type: ignore[no-any-return]


def capabilities(studio: Any) -> dict[str, Any]:
    """What the loaded model can do, as far as the engine says: /props has `modalities` and `chat_template_caps`."""
    props = dict(studio.engine.props or {})
    modalities = props.get("modalities") if isinstance(props.get("modalities"), dict) else {}
    caps = props.get("chat_template_caps") if isinstance(props.get("chat_template_caps"), dict) else {}
    return {
        "vision": bool(modalities.get("vision")),
        "tools": caps.get("supports_tools") if "supports_tools" in caps else None,  # None: the engine did not say
        "n_ctx": studio.engine.status().get("n_ctx"),
        "max_upload_mb": upload_limit_mb(studio),
        "pdf": importlib.util.find_spec("pypdf") is not None,
    }


def upload_limit_mb(studio: Any) -> int:
    try:
        return max(1, min(512, int(studio.config.get("chat.max_upload_mb"))))
    except (TypeError, ValueError):
        return 25


def screen_proxy_body(path: str, body: bytes | None) -> None:
    """Refuse a chat request that asks for tools and for a forced answer format together (see tools.py).
    Raises ValueError, which the server answers with 400 and the message."""
    if not body or not path.split("?", 1)[0].rstrip("/").endswith("/chat/completions") or b'"tools"' not in body:
        return
    try:
        doc = json.loads(body.decode("utf-8"))
    except ValueError:
        return  # the engine will say what is wrong with it
    if isinstance(doc, dict):
        tools.check_tools_and_format(doc)


# ------------------------------------------------------------------------------------------------ handlers


def api_chat_caps(self: Any, query: dict[str, str]) -> None:
    self.send_json(200, capabilities(self.studio))


def api_attachment_upload(self: Any, query: dict[str, str]) -> None:
    studio = self.studio
    limit_mb = upload_limit_mb(studio)
    if self._unread > limit_mb * 1024 * 1024:
        raise _fail(
            413,
            f"This file is {self._unread / 1048576:.1f} MB; the limit is {limit_mb} MB "
            "(Settings → Chat → Largest upload).",
        )
    name = (query.get("name") or "file").replace("\\", "/").rsplit("/", 1)[-1][:200] or "file"
    data = self._body(limit_mb * 1024 * 1024)
    try:
        result = attachments.extract(name, data)
    except attachments.AttachmentError as exc:
        raise _fail(exc.status, str(exc)) from None
    if result["kind"] == "image":
        caps = capabilities(studio)
        if not caps["vision"]:
            raise _fail(
                409,
                "The loaded model cannot look at pictures. Load a vision model (a GGUF with its "
                "mmproj file) to attach them."
                if studio.engine.props
                else "Load a model first. Pictures only work with a model that can look at them.",
            )
        blob = studio.chats.put_image(data, result["ext"])
        result = {
            "kind": "image",
            "name": name,
            "mime": result["mime"],
            "blob": blob,
            "size": result["size"],
            "tokens": result["tokens"],
        }
    self.send_json(200, result)


def api_attachment_get(self: Any, query: dict[str, str], name: str) -> None:
    data, mime = self.studio.chats.read_image(name)
    self._send(
        200, data, mime, {"Content-Security-Policy": "default-src 'none'", "Cache-Control": "private, max-age=3600"}
    )


def api_chat_search(self: Any, query: dict[str, str]) -> None:
    self.send_json(200, {"results": self.studio.chats.search(query.get("q", ""))})


def _file_name(chat: dict[str, Any], ext: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(chat.get("title") or "chat")).strip("-")[:50] or "chat"
    return f"{slug}-{str(chat.get('id', ''))[:6]}.{ext}"


def api_chat_export(self: Any, query: dict[str, str], id: str) -> None:  # noqa: A002
    chat = self.studio.chats.get(id)
    fmt = query.get("format", "json")
    if fmt == "md":
        body = chat_tree.export_markdown(chat, __version__).encode("utf-8")
        ctype, ext = "text/markdown; charset=utf-8", "md"
    elif fmt == "json":
        doc = chat_tree.export_json(chat, __version__, images=self.studio.chats.image_data_url)
        body = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
        ctype, ext = "application/json; charset=utf-8", "json"
    else:
        raise _fail(400, "format must be json or md")
    name = _file_name(chat, ext)
    self._send(
        200, body, ctype, {"Content-Disposition": f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}"}
    )


def api_chat_import(self: Any, query: dict[str, str]) -> None:
    limit = upload_limit_mb(self.studio) * 1024 * 1024
    if self._unread > limit:
        raise _fail(
            413,
            f"This file is {self._unread / 1048576:.1f} MB; the limit is {upload_limit_mb(self.studio)} MB "
            "(Settings → Chat → Largest upload).",
        )
    text = self._body(limit).decode("utf-8-sig", "replace")
    fmt = query.get("format") or ("json" if text.lstrip()[:1] in ("{", "[") else "md")
    try:
        doc = chat_tree.import_json(text) if fmt == "json" else chat_tree.import_markdown(text)
        chat = self.studio.chats.import_chat(doc)
    except chat_tree.ChatImportError as exc:
        raise _fail(400, str(exc)) from None
    self.send_json(200, {"id": chat["id"], "title": chat["title"], "messages": len(chat["messages"])})


def api_tools(self: Any, query: dict[str, str]) -> None:
    self.send_json(200, toolbox(self.studio).status())


def api_tools_call(self: Any, query: dict[str, str]) -> None:
    body = self.read_json()
    box = toolbox(self.studio)
    try:
        result = box.call(
            str(body.get("name") or ""), body.get("arguments") or {}, approved=body.get("approved") is True
        )
    except tools.NeedsApproval:
        raise _fail(409, "This tool needs your approval first.") from None
    except KeyError:
        raise _fail(404, "That tool is not available (turned off, or its server is not connected).") from None
    self.send_json(200, result)


def api_tools_reload(self: Any, query: dict[str, str]) -> None:
    box = toolbox(self.studio)
    box.sync_mcp()
    box.mcp.reconnect()
    box.mcp.wait_idle(1.0)
    self.send_json(200, box.status())


def install(handler: type, routes: list[Any], api_error: type) -> None:
    """Add the handlers to the server's Handler class and the routes to its route table."""
    global _ApiError
    _ApiError = api_error
    for method, pattern, name in ROUTES:
        routes.append((method, re.compile(pattern), name))
        setattr(handler, f"api_{name}", globals()[f"api_{name}"])
