"""Branching conversations.

A chat file keeps every message of the conversation as a node: `id`, `parent` (None for a first message) and the usual
fields. `leaf` names the last message of the branch that is shown; the path from that node up to the root is the
conversation as the model sees it. Editing a message or asking again adds a sibling instead of overwriting, and the
page switches between siblings ("2 / 3").

Older files hold a plain list without ids; `normalize` turns that into a chain, so nothing is lost.

This module also turns a chat into JSON or Markdown and back, and searches a chat's text.
"""

from __future__ import annotations

import json
import re
import secrets
import time
from typing import Any

FORMAT = "fle-studio-chat"
VERSION = 2
ROLES = ("user", "assistant", "system", "tool")
_ID = re.compile(r"^[\w-]{1,40}$")
_ESCAPED_HEADING = re.compile(r"^\\(## (?:User|Assistant|System)\b)", re.M)
_HEADING = re.compile(r"^## (User|Assistant|System)\b")


def new_id() -> str:
    return secrets.token_hex(6)


def _flatten(content: Any) -> str:
    """The text of a message body that may be a string or a list of OpenAI-style parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def normalize(messages: list[Any], leaf: Any = None) -> tuple[list[dict[str, Any]], str | None]:
    """Messages with unique ids and valid parents, and a valid leaf. Plain lists become one chain."""
    nodes = [dict(m) for m in messages if isinstance(m, dict)]
    legacy = not any("parent" in m for m in nodes)
    seen: set[str] = set()
    for m in nodes:
        ident = m.get("id")
        if not isinstance(ident, str) or not _ID.match(ident) or ident in seen:
            ident = new_id()
            while ident in seen:
                ident = new_id()
            m["id"] = ident
        seen.add(ident)
    prev: str | None = None
    for m in nodes:
        if legacy:
            m["parent"] = prev
        else:
            parent = m.get("parent")
            m["parent"] = parent if isinstance(parent, str) and parent in seen and parent != m["id"] else None
        prev = m["id"]
    # A cycle would make the path endless: cut any node that is its own ancestor.
    by_id = {m["id"]: m for m in nodes}
    for m in nodes:
        walk, hops = m, 0
        while walk.get("parent") and hops <= len(nodes):
            walk = by_id[walk["parent"]]
            hops += 1
        if hops > len(nodes):
            m["parent"] = None
    if not isinstance(leaf, str) or leaf not in by_id:
        leaf = nodes[-1]["id"] if nodes else None
    return nodes, leaf


def active_path(messages: list[dict[str, Any]], leaf: str | None) -> list[dict[str, Any]]:
    """The conversation shown: from the root down to the leaf."""
    by_id = {m.get("id"): m for m in messages}
    out: list[dict[str, Any]] = []
    node = by_id.get(leaf)
    while node is not None and len(out) <= len(messages):
        out.append(node)
        node = by_id.get(node.get("parent"))
    out.reverse()
    return out


def message_text(m: dict[str, Any]) -> str:
    return _flatten(m.get("content"))


# ------------------------------------------------------------------------------------------------ search


def search_text(chat: dict[str, Any]) -> str:
    """Everything of a chat that a search looks through (all branches, attachment names and text)."""
    parts = [str(chat.get("title") or "")]
    for m in chat.get("messages") or []:
        if not isinstance(m, dict):
            continue
        parts.append(message_text(m))
        for a in m.get("attachments") or []:
            if isinstance(a, dict):
                parts.append(str(a.get("name") or ""))
                if isinstance(a.get("text"), str):
                    parts.append(a["text"])
        for call in m.get("tool_calls") or []:
            if isinstance(call, dict):
                parts.append(str(call.get("name") or ""))
    return "\n".join(parts)


def snippet(text: str, needle: str, width: int = 80) -> str:
    """A short piece of `text` around the first hit of `needle` (both compared without case)."""
    flat = " ".join(text.split())
    at = flat.casefold().find(needle.casefold())
    if at < 0:
        return flat[: width * 2]
    start = max(0, at - width // 2)
    end = min(len(flat), at + len(needle) + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


# ------------------------------------------------------------------------------------------------ export


def export_json(chat: dict[str, Any], app_version: str, images: Any = None) -> dict[str, Any]:
    """The chat as a portable document. `images(blob_name) -> data URL | None` inlines pictures."""
    messages = []
    for m in chat.get("messages") or []:
        m = dict(m)
        atts = []
        for a in m.get("attachments") or []:
            a = dict(a)
            if a.get("kind") == "image" and a.get("blob") and images:
                a["data"] = images(a["blob"])
                a.pop("blob", None)
            atts.append(a)
        if atts:
            m["attachments"] = atts
        messages.append(m)
    return {
        "format": FORMAT,
        "version": VERSION,
        "exported": time.time(),
        "app_version": app_version,
        "chat": {
            "title": chat.get("title") or "",
            "created": chat.get("created"),
            "updated": chat.get("updated"),
            "model": chat.get("model") or "",
            "system": chat.get("system") or "",
            "leaf": chat.get("leaf"),
            "messages": messages,
        },
    }


def _escape(text: str) -> str:
    return re.sub(r"^(## (?:User|Assistant|System)\b)", r"\\\1", text, flags=re.M)


def export_markdown(chat: dict[str, Any], app_version: str) -> str:
    """The shown branch as Markdown. Reasoning and other branches stay in the JSON export."""
    path = active_path(chat.get("messages") or [], chat.get("leaf"))
    lines = [f"# {chat.get('title') or 'Chat'}", ""]
    meta = [f"Exported from FLE Studio {app_version}"]
    if chat.get("model"):
        meta.append(f"model {chat['model']}")
    lines += ["> " + " · ".join(meta), ""]
    if chat.get("system"):
        lines += ["## System", "", _escape(str(chat["system"])), ""]
    for m in path:
        role = m.get("role")
        if role == "tool":
            continue  # a tool result is shown with the call that asked for it
        if role not in ("user", "assistant", "system"):
            continue
        lines += [f"## {role.capitalize()}", ""]
        for a in m.get("attachments") or []:
            if isinstance(a, dict):
                what = f"{a.get('chars')} characters" if a.get("kind") == "text" else "image"
                lines += [f"*Attachment: {a.get('name') or 'file'} ({what})*", ""]
        text = message_text(m).strip()
        if text:
            lines += [_escape(text), ""]
        for call in m.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            lines += [f"**Tool call:** `{call.get('name')}` `{_one_line(call.get('arguments'))}`", ""]
            result = call.get("result")
            if isinstance(result, str) and result:
                fence = "````" if "```" in result else "```"
                lines += [f"{fence}text", _escape(result), fence, ""]
    return "\n".join(lines).rstrip() + "\n"


def _one_line(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return " ".join(text.split()).replace("`", "'")[:300]


# ------------------------------------------------------------------------------------------------ import


class ChatImportError(ValueError):
    """The file is not a chat this version can read."""


def _plain_messages(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant", "system"):
            continue
        text = _flatten(item.get("content"))
        if not text and not item.get("reasoning_content"):
            continue
        msg: dict[str, Any] = {"role": item["role"], "content": text}
        if isinstance(item.get("reasoning_content"), str):
            msg["reasoning"] = item["reasoning_content"]
        out.append(msg)
    return out


def import_json(text: str) -> dict[str, Any]:
    """A chat dict (no id yet) from an export of this app, an OpenAI-style {"messages": [...]} or a bare list."""
    try:
        doc = json.loads(text)
    except ValueError:
        raise ChatImportError("That file is not valid JSON") from None
    body: Any = doc
    if isinstance(doc, dict) and doc.get("format") == FORMAT:
        body = doc.get("chat")
        if not isinstance(body, dict):
            raise ChatImportError("The export has no chat in it")
        version = doc.get("version")
        if isinstance(version, int) and version > VERSION:
            raise ChatImportError(f"This export is from a newer FLE Studio (format version {version})")
        messages = [m for m in body.get("messages") or [] if isinstance(m, dict) and m.get("role") in ROLES]
        return {
            "title": str(body.get("title") or ""),
            "model": str(body.get("model") or ""),
            "system": str(body.get("system") or ""),
            "created": body.get("created"),
            "leaf": body.get("leaf"),
            "messages": messages,
        }
    if isinstance(doc, dict) and isinstance(doc.get("messages"), list):
        messages = _plain_messages(doc["messages"])
        title = str(doc.get("title") or "")
    elif isinstance(doc, list):
        messages, title = _plain_messages(doc), ""
    else:
        raise ChatImportError("That JSON is not a chat: expected an export from FLE Studio or a list of messages")
    if not messages:
        raise ChatImportError("The file holds no messages")
    system = ""
    if messages and messages[0]["role"] == "system":
        system = messages.pop(0)["content"]
    return {"title": title, "model": "", "system": system, "messages": messages}


def import_markdown(text: str) -> dict[str, Any]:
    """Read back what `export_markdown` writes: a title, then `## User` / `## Assistant` sections."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = ""
    sections: list[tuple[str, list[str]]] = []
    in_fence = ""
    for line in lines:
        fence = re.match(r"^(`{3,}|~{3,})", line)
        if fence and not in_fence:
            in_fence = fence.group(1)
        elif fence and in_fence and line.startswith(in_fence) and not line[len(in_fence) :].strip(" `~"):
            in_fence = ""
        head = _HEADING.match(line) if not in_fence else None
        if head:
            sections.append((head.group(1).lower(), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.startswith("# ") and not title:
            title = line[2:].strip()
    messages: list[dict[str, Any]] = []
    system = ""
    for role, body in sections:
        content = [ln for ln in body if not re.match(r"^\*Attachment: .*\*$", ln)]
        text_ = _ESCAPED_HEADING.sub(r"\1", "\n".join(content)).strip()
        if role == "system" and not messages:
            system = text_
        elif text_:
            messages.append({"role": role, "content": text_})
    if not messages:
        raise ChatImportError("No '## User' or '## Assistant' sections found in that Markdown file")
    return {"title": title, "model": "", "system": system, "messages": messages}
