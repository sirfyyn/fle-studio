"""Live: everything that goes through the /v1 address of FLE Studio, as a temporary chat that only lives in memory.

Every chat completion that passes the proxy (the chat page and any other client that talks to the /v1 address) is added
here: the messages of the request that are new, and the answer. Requests that continue a conversation (the earlier
messages are the same) are put together into one thread, so a client that sends the whole history each time shows up as
one conversation, not as a pile of requests. Pictures are left out, only a note that there was one.

Nothing is written to disk. At most MAX_MESSAGES messages are kept; the oldest fall away first. To keep a conversation,
star it (or one of its messages): that copies it, up to that message, into the normal chat archive, where it stays.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from . import chat_tree
from .capture import _strip_images

MAX_MESSAGES = 100
TITLE_CHARS = 60


def _text(content: Any) -> str:
    return chat_tree._flatten(content)


def _sig(role: Any, content: Any) -> tuple[str, str]:
    """What two messages are compared by: the role and the text (a picture note counts as its text)."""
    return (str(role or ""), _text(content).strip())


class LiveChat:
    def __init__(self, max_messages: int = MAX_MESSAGES) -> None:
        self.max_messages = max(1, int(max_messages))
        self._lock = threading.Lock()
        self._threads: list[dict[str, Any]] = []
        self._version = 0  # counts every change, so a page can tell that it has to fetch again

    # ------------------------------------------------------------------------------------------ adding

    def add(self, record: dict[str, Any]) -> None:
        """One finished request as the capture builds it (keys: anfrage, antwort, status, zeit). Never raises."""
        try:
            self._add(record)
        except Exception:  # showing a conversation must never break the answer
            pass

    def _add(self, record: dict[str, Any]) -> None:
        if int(record.get("status") or 0) != 200:
            return
        request = record.get("anfrage") or {}
        answer = record.get("antwort") or {}
        if not isinstance(request, dict) or not isinstance(answer, dict):
            return
        sent = self._request_messages(request)
        reply = self._reply_message(answer)
        if not sent or reply is None:
            return
        now = time.time()
        model = str(answer.get("model") or request.get("model") or "")
        with self._lock:
            thread = self._find(sent)
            if thread is None:
                thread = {"id": chat_tree.new_id(), "messages": [], "dropped": 0, "sig": [], "created": now}
                self._threads.append(thread)
                new = sent
            else:
                new = sent[thread["dropped"] + len(thread["messages"]) :]
            for m in new:
                self._push(thread, {**m, "t": now})
            reply["model"] = model
            reply["t"] = now
            self._push(thread, reply)
            thread["updated"] = now
            thread["model"] = model
            self._trim()
            self._version += 1

    def _request_messages(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        messages = request.get("messages")
        if isinstance(messages, list):
            for m in messages:
                if not isinstance(m, dict) or m.get("role") not in chat_tree.ROLES:
                    continue
                msg: dict[str, Any] = {"role": m["role"], "content": _strip_images(m.get("content", ""))}
                if m.get("tool_calls"):
                    msg["tool_calls"] = _strip_images(m["tool_calls"])
                if m.get("tool_call_id"):
                    msg["tool_call_id"] = m["tool_call_id"]
                if m.get("reasoning_content"):
                    msg["reasoning"] = str(m["reasoning_content"])
                out.append(msg)
        elif isinstance(request.get("prompt"), str):
            out.append({"role": "user", "content": request["prompt"]})
        return out

    def _reply_message(self, answer: dict[str, Any]) -> dict[str, Any] | None:
        choices = answer.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        if isinstance(choice.get("message"), dict):
            body = choice["message"]
            msg: dict[str, Any] = {"role": "assistant", "content": _strip_images(body.get("content") or "")}
            if body.get("reasoning_content"):
                msg["reasoning"] = str(body["reasoning_content"])
            if body.get("tool_calls"):
                msg["tool_calls"] = _strip_images(body["tool_calls"])
        elif isinstance(choice.get("text"), str):
            msg = {"role": "assistant", "content": choice["text"]}
        else:
            return None
        timings = answer.get("timings")
        if isinstance(timings, dict):
            speed = {
                k: timings[k]
                for k in ("predicted_per_second", "prompt_per_second", "draft_n", "draft_n_accepted")
                if isinstance(timings.get(k), (int, float))
            }
            if speed:
                msg["timings"] = speed
        if choice.get("finish_reason"):
            msg["finish"] = choice["finish_reason"]
        return msg

    def _find(self, sent: list[dict[str, Any]]) -> dict[str, Any] | None:
        """The thread whose messages are the start of this request: the longest such one, the newest on a tie."""
        mine = [_sig(m["role"], m.get("content")) for m in sent]
        best: dict[str, Any] | None = None
        for thread in self._threads:
            start, held = thread["dropped"], thread["sig"]
            if len(mine) < start + len(held) or not held:
                continue
            if mine[start : start + len(held)] != held:
                continue
            if best is None or len(held) >= len(best["sig"]):
                best = thread
        return best

    @staticmethod
    def _push(thread: dict[str, Any], msg: dict[str, Any]) -> None:
        msg = {"id": chat_tree.new_id(), **msg}
        thread["messages"].append(msg)
        thread["sig"].append(_sig(msg["role"], msg.get("content")))

    def _total(self) -> int:
        return sum(len(t["messages"]) for t in self._threads)

    def _trim(self) -> None:
        """Drop the oldest messages until at most max_messages are left; a thread that is empty goes away."""
        while self._total() > self.max_messages:
            oldest = min(self._threads, key=lambda t: t["updated"])
            oldest["messages"].pop(0)
            oldest["sig"].pop(0)
            oldest["dropped"] += 1
            if not oldest["messages"]:
                self._threads.remove(oldest)

    # ------------------------------------------------------------------------------------------ reading

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            threads = sorted(self._threads, key=lambda t: t["updated"], reverse=True)
            return {
                "max_messages": self.max_messages,
                "version": self._version,
                "count": self._total(),
                "threads": [self._public(t) for t in threads],
            }

    def summary(self) -> dict[str, Any]:
        """Only what the page polls to know whether to fetch the conversations again."""
        with self._lock:
            return {
                "count": self._total(),
                "threads": len(self._threads),
                "updated": max((t["updated"] for t in self._threads), default=0),
                "version": self._version,
            }

    def _public(self, thread: dict[str, Any]) -> dict[str, Any]:
        first = next((_text(m.get("content")) for m in thread["messages"] if m["role"] == "user"), "")
        return {
            "id": thread["id"],
            "title": " ".join(first.split())[:TITLE_CHARS] or "Request",
            "model": thread.get("model", ""),
            "updated": thread["updated"],
            "cut": thread["dropped"] > 0,
            "starred": bool(thread.get("starred")),
            "messages": json.loads(json.dumps(thread["messages"])),
        }

    # ------------------------------------------------------------------------------------------ changing

    def delete(self, ident: str) -> bool:
        with self._lock:
            for t in self._threads:
                if t["id"] == ident:
                    self._threads.remove(t)
                    self._version += 1
                    return True
        return False

    def clear(self) -> int:
        with self._lock:
            n = len(self._threads)
            self._threads.clear()
            self._version += 1
        return n

    def star_data(self, ident: str, message: str | None = None) -> dict[str, Any]:
        """A chat (the shape chats.save takes) from a thread, up to `message` if one is named. KeyError if gone."""
        with self._lock:
            thread = next((t for t in self._threads if t["id"] == ident), None)
            if thread is None:
                raise KeyError(ident)
            messages = [dict(m) for m in thread["messages"]]
            model = thread.get("model", "")
            for m in thread["messages"]:
                if m["id"] == message:
                    m["starred"] = True
                    self._version += 1
        if message is not None:
            ids = [m["id"] for m in messages]
            if message not in ids:
                raise KeyError(message)
            messages = messages[: ids.index(message) + 1]
        chain: list[dict[str, Any]] = []
        prev: str | None = None
        for m in messages:
            node = {k: v for k, v in m.items() if k not in ("timings", "finish", "starred")}
            node["parent"] = prev
            prev = node["id"]
            chain.append(node)
        return {"messages": chain, "leaf": prev, "model": model, "favorite": True}

    def mark_starred(self, ident: str) -> None:
        """Remember in the thread that it was saved, so the page can show it."""
        with self._lock:
            for t in self._threads:
                if t["id"] == ident:
                    t["starred"] = True
                    self._version += 1
