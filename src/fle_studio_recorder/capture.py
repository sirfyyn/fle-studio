"""Capture: keep what you do with the model, as plain files on this computer, to build your own corpus.

Off by default. When it is on, every chat completion that goes through FLE Studio (the chat page and any other
client that talks to the /v1 address) is written as one line of JSON to <data folder>/capture/YYYY-MM-DD.jsonl:
the request, the answer in the usual chat-completion shape (a streamed answer is put back together), the engine's
timings (speed, draft tokens accepted) and how long it took. Nothing is sent anywhere. The lines look like the ones
a corpus for MTP-head training, imatrix calibration or a test set needs. Pictures are not kept, only a note that
there was one.

The files are readable by you only. A size limit stops the recording instead of filling the disk.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

SUBDIR = "capture"
RECORDED = ("/chat/completions", "/completions")
IMAGE_NOTE = "[image removed: {n} characters]"


def _strip_images(obj: Any) -> Any:
    """Replace inline data: URLs (pictures, files) by a short note; everything else is kept as it is."""
    if isinstance(obj, dict):
        return {k: _strip_images(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_images(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:") and len(obj) > 256:
        return IMAGE_NOTE.format(n=len(obj))
    return obj


class Recorder:
    """One request in flight. Feed it the answer as it arrives, then call done()."""

    def __init__(self, owner: Capture, path: str, request: dict[str, Any]) -> None:
        self.owner = owner
        self.path = path
        self.request = request
        self.started = time.time()
        self.stream = bool(request.get("stream"))
        self._buf = b""
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._tools: dict[int, dict[str, Any]] = {}
        self._finish: str | None = None
        self._timings: dict[str, Any] | None = None
        self._usage: dict[str, Any] | None = None
        self._meta: dict[str, Any] = {}

    @property
    def timings(self) -> dict[str, Any] | None:
        """The engine's timings of a streamed answer, once it has been read (for the speed shown on the Live panel)."""
        return self._timings

    @property
    def usage(self) -> dict[str, Any] | None:
        return self._usage

    def wants_stream(self) -> bool:
        return self.owner.wants_stream()

    def feed(self, chunk: bytes) -> None:
        """A piece of a text/event-stream answer."""
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.decode("utf-8", "replace").strip())

    def _line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if not data or data == "[DONE]":
            return
        try:
            obj = json.loads(data)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in obj and key not in self._meta:
                self._meta[key] = obj[key]
        if obj.get("timings"):
            self._timings = obj["timings"]
        if obj.get("usage"):
            self._usage = obj["usage"]
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self._content.append(delta["content"])
            if delta.get("reasoning_content"):
                self._reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                empty = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                slot = self._tools.setdefault(int(call.get("index", 0)), empty)
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                self._finish = choice["finish_reason"]

    def done(self, status: int, body: bytes | None = None) -> None:
        """The answer is complete (or the browser stopped reading). body: the whole answer if not streamed."""
        try:
            if self.stream:
                if self._buf.strip():
                    self._line(self._buf.decode("utf-8", "replace").strip())
                message: dict[str, Any] = {"role": "assistant", "content": "".join(self._content)}
                if self._reasoning:
                    message["reasoning_content"] = "".join(self._reasoning)
                if self._tools:
                    message["tool_calls"] = [self._tools[i] for i in sorted(self._tools)]
                answer: dict[str, Any] = {
                    **self._meta,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": self._finish}],
                }
                if self._usage:
                    answer["usage"] = self._usage
                if self._timings:
                    answer["timings"] = self._timings
            else:
                try:
                    answer = json.loads((body or b"").decode("utf-8"))
                except ValueError:
                    answer = {}
            self.owner.record(
                {
                    "zeit": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started)),
                    "pfad": self.path.split("?", 1)[0],
                    "anfrage": _strip_images(self.request),
                    "antwort": _strip_images(answer),
                    "status": status,
                    "dauer_ms": int((time.time() - self.started) * 1000),
                    "stream": self.stream,
                    "quelle": "fle-studio",
                }
            )
        except Exception:  # recording must never break the answer the user is waiting for
            pass


class Capture:
    def __init__(
        self,
        data_dir: Path,
        enabled: Any,
        max_mb: Any,
        live: Any = None,
        watched: Any = lambda: False,
    ) -> None:
        self.dir = Path(data_dir) / SUBDIR
        self._enabled = enabled  # callables, so a change in the settings takes effect at once
        self._max_mb = max_mb
        self.live = live  # the in-memory Live chat (livechat.py), which sees every request whether or not Capture is on
        self._watched = watched  # is a page open that may show the Live chat?
        self._lock = threading.Lock()

    def wants_stream(self) -> bool:
        """Whether a streamed answer is read at all: Capture is on, or somebody can look at the Live chat. Otherwise
        the proxy only passes the bytes on."""
        return bool(self._enabled()) or bool(self._watched())

    def begin(self, path: str, body: bytes | None) -> Recorder | None:
        if not (self._enabled() or self.live) or not body or not path.split("?", 1)[0].rstrip("/").endswith(RECORDED):
            return None
        try:
            request = json.loads(body.decode("utf-8"))
        except ValueError:
            return None
        if not isinstance(request, dict):
            return None
        return Recorder(self, path, request)

    def size(self) -> int:
        try:
            return sum(p.stat().st_size for p in self.dir.glob("*.jsonl"))
        except OSError:
            return 0

    def record(self, record: dict[str, Any]) -> None:
        """A finished request: to the Live chat (memory), and to the files when Capture is on."""
        if self.live:
            self.live.add(record)
        if self._enabled():
            self.write(record)

    def write(self, record: dict[str, Any]) -> bool:
        limit = int(self._max_mb() or 0) * 1024 * 1024
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            if limit and self.size() + len(line.encode("utf-8")) > limit:
                return False
            self.dir.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(self.dir, 0o700)
            except OSError:
                pass
            path = self.dir / (time.strftime("%Y-%m-%d", time.localtime()) + ".jsonl")
            fresh = not path.exists()
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            if fresh:
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
        return True

    def status(self) -> dict[str, Any]:
        files = sorted(self.dir.glob("*.jsonl")) if self.dir.is_dir() else []
        count = 0
        for p in files:
            try:
                with open(p, "rb") as f:
                    count += sum(1 for _ in f)
            except OSError:
                pass
        return {
            "enabled": bool(self._enabled()),
            "folder": str(self.dir),
            "files": len(files),
            "records": count,
            "bytes": self.size(),
            "max_mb": int(self._max_mb() or 0),
            "first": files[0].stem if files else "",
            "last": files[-1].stem if files else "",
        }

    def clear(self) -> int:
        """Delete every captured file; returns how many."""
        n = 0
        with self._lock:
            for p in self.dir.glob("*.jsonl") if self.dir.is_dir() else []:
                try:
                    p.unlink()
                    n += 1
                except OSError:
                    pass
        return n
