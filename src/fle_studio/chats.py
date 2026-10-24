"""Conversations, one JSON file each in <data>/chats. Nothing leaves the machine.

A chat holds all its messages as a tree (see chat_tree.py). Pictures that were attached are kept next to the chats
in <data>/chats/files, named by their SHA-256, so a chat file stays small.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from . import chat_tree, paths

_ID = re.compile(r"^[0-9a-f]{16}$")
_BLOB = re.compile(r"^[0-9a-f]{64}\.(png|jpg|gif|webp)$")
MAX_BYTES = 32 * 1024 * 1024
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "gif": "image/gif", "webp": "image/webp"}


class Chats:
    def __init__(self, data_dir: Path) -> None:
        self.dir = paths.ensure(data_dir / "chats")
        self.files = paths.ensure(self.dir / "files")
        self._lock = threading.Lock()

    def _path(self, ident: str) -> Path:
        if not _ID.match(ident):
            raise KeyError(ident)
        return self.dir / f"{ident}.json"

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in self.dir.glob("*.json"):
            d = self._read(p)
            if d is None:
                continue
            nodes, leaf = chat_tree.normalize(d.get("messages") or [], d.get("leaf"))
            out.append(
                {
                    "id": d.get("id", p.stem),
                    "title": d.get("title") or "New chat",
                    "updated": d.get("updated", 0),
                    "model": d.get("model", ""),
                    "messages": len(chat_tree.active_path(nodes, leaf)),
                }
            )
        out.sort(key=lambda c: c["updated"], reverse=True)
        return out

    def get(self, ident: str) -> dict[str, Any]:
        data = self._read(self._path(ident))
        if data is None:
            raise KeyError(ident)
        data["messages"], data["leaf"] = chat_tree.normalize(data.get("messages") or [], data.get("leaf"))
        return data

    def save(self, ident: str | None, data: dict[str, Any]) -> dict[str, Any]:
        ident = ident or secrets.token_hex(8)
        path = self._path(ident)
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        messages, leaf = chat_tree.normalize(messages, data.get("leaf"))
        shown = chat_tree.active_path(messages, leaf)
        now = time.time()
        title = str(data.get("title") or "").strip()
        if not title:
            first = next((chat_tree.message_text(m) for m in shown if m.get("role") == "user"), "")
            title = " ".join(first.split())[:60] or "New chat"
        chat = {
            "id": ident,
            "version": chat_tree.VERSION,
            "title": title[:120],
            "created": data.get("created") or now,
            "updated": now,
            "model": str(data.get("model") or ""),
            "system": str(data.get("system") or ""),
            "leaf": leaf,
            "messages": messages,
        }
        text = json.dumps(chat, ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_BYTES:
            raise ValueError("chat too large")
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
        return chat

    def delete(self, ident: str) -> None:
        try:
            self._path(ident).unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------------------------------ search

    def search(self, query: str, limit: int = 40) -> list[dict[str, Any]]:
        needle = query.strip()
        if not needle:
            return []
        out = []
        for p in self.dir.glob("*.json"):
            d = self._read(p)
            if d is None:
                continue
            text = chat_tree.search_text(d)
            if needle.casefold() not in text.casefold():
                continue
            title = str(d.get("title") or "New chat")
            in_title = needle.casefold() in title.casefold()
            out.append(
                {
                    "id": d.get("id", p.stem),
                    "title": title,
                    "updated": d.get("updated", 0),
                    "snippet": chat_tree.snippet(text, needle) if not in_title else "",
                    "in_title": in_title,
                }
            )
        out.sort(key=lambda c: (not c["in_title"], -float(c["updated"] or 0)))
        return out[:limit]

    # ------------------------------------------------------------------------------------------ pictures

    def put_image(self, data: bytes, ext: str) -> str:
        """Store a picture and return its name `<sha256>.<ext>`."""
        if ext not in IMAGE_TYPES:
            raise ValueError("unsupported picture type")
        name = f"{hashlib.sha256(data).hexdigest()}.{ext}"
        path = self.files / name
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        return name

    def read_image(self, name: str) -> tuple[bytes, str]:
        if not _BLOB.match(name):
            raise KeyError(name)
        try:
            return (self.files / name).read_bytes(), IMAGE_TYPES[name.rsplit(".", 1)[1]]
        except FileNotFoundError:
            raise KeyError(name) from None

    def image_data_url(self, name: str) -> str | None:
        try:
            data, mime = self.read_image(name)
        except KeyError:
            return None
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def store_data_url(self, url: str) -> str | None:
        """Keep a `data:image/...;base64,...` URL as a file; returns its name, or None when it is not a picture."""
        m = re.match(r"^data:image/(png|jpeg|gif|webp);base64,([A-Za-z0-9+/=\s]+)$", url or "")
        if not m:
            return None
        try:
            data = base64.b64decode(m.group(2), validate=False)
        except ValueError:
            return None
        return self.put_image(data, "jpg" if m.group(1) == "jpeg" else m.group(1))

    # ------------------------------------------------------------------------------------------ import

    def import_chat(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Save an imported chat under a new id. Inline pictures become files again."""
        messages = []
        for m in doc.get("messages") or []:
            m = dict(m)
            atts = []
            for a in m.get("attachments") or []:
                if not isinstance(a, dict):
                    continue
                a = dict(a)
                if a.get("kind") == "image":
                    name = self.store_data_url(str(a.pop("data", "") or ""))
                    if not name:
                        continue
                    a["blob"] = name
                atts.append(a)
            if atts:
                m["attachments"] = atts
            else:
                m.pop("attachments", None)
            messages.append(m)
        return self.save(None, {**doc, "messages": messages})
