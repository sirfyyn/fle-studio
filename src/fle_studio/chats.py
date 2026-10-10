"""Conversations, one JSON file each in <data>/chats. Nothing leaves the machine."""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from . import paths

_ID = re.compile(r"^[0-9a-f]{16}$")
MAX_BYTES = 32 * 1024 * 1024


class Chats:
    def __init__(self, data_dir: Path) -> None:
        self.dir = paths.ensure(data_dir / "chats")
        self._lock = threading.Lock()

    def _path(self, ident: str) -> Path:
        if not _ID.match(ident):
            raise KeyError(ident)
        return self.dir / f"{ident}.json"

    def list(self) -> list[dict[str, Any]]:
        out = []
        for p in self.dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"id": d.get("id", p.stem), "title": d.get("title") or "New chat",
                        "updated": d.get("updated", 0), "model": d.get("model", ""),
                        "messages": len(d.get("messages", []))})
        out.sort(key=lambda c: c["updated"], reverse=True)
        return out

    def get(self, ident: str) -> dict[str, Any]:
        try:
            return json.loads(self._path(ident).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise KeyError(ident) from None

    def save(self, ident: str | None, data: dict[str, Any]) -> dict[str, Any]:
        ident = ident or secrets.token_hex(8)
        path = self._path(ident)
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        now = time.time()
        title = str(data.get("title") or "").strip()
        if not title:
            first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
            title = " ".join(str(first).split())[:60] or "New chat"
        chat = {
            "id": ident,
            "title": title[:120],
            "created": data.get("created") or now,
            "updated": now,
            "model": str(data.get("model") or ""),
            "system": str(data.get("system") or ""),
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
