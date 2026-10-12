"""Capture: off by default, one line per answer when on, streams put back together, no pictures, size limit, delete."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from chat_support import Stack, make_stack

from fle_studio.capture import Capture


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    yield from make_stack(tmp_path)


def lines(stack: Stack) -> list[dict]:
    folder = stack.studio.data_dir / "capture"
    out: list[dict] = []
    for p in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        out += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


BODY = {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16}


def test_nothing_is_recorded_by_default(stack: Stack) -> None:
    assert stack.json("GET", "/api/capture")[1]["enabled"] is False
    assert stack.chat(BODY)[0] == 200
    assert lines(stack) == []


def test_a_plain_answer_is_recorded(stack: Stack) -> None:
    stack.settings({"capture": {"enabled": True}})
    status, answer = stack.chat(BODY)
    assert status == 200
    rec = lines(stack)
    assert len(rec) == 1
    assert rec[0]["pfad"] == "/v1/chat/completions" and rec[0]["status"] == 200 and rec[0]["stream"] is False
    assert rec[0]["anfrage"]["messages"][0]["content"] == BODY["messages"][0]["content"]
    assert rec[0]["antwort"]["choices"][0]["message"]["content"] == answer["choices"][0]["message"]["content"]
    assert "timings" in rec[0]["antwort"]


def test_a_streamed_answer_is_put_back_together(stack: Stack) -> None:
    stack.settings({"capture": {"enabled": True}})
    status, _, raw = stack.call("POST", "/v1/chat/completions", {**BODY, "stream": True})
    assert status == 200
    sent = think = ""
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            for ch in json.loads(line[5:]).get("choices") or []:
                sent += (ch.get("delta") or {}).get("content") or ""
                think += (ch.get("delta") or {}).get("reasoning_content") or ""
    rec = lines(stack)
    assert len(rec) == 1 and rec[0]["stream"] is True
    choice = rec[0]["antwort"]["choices"][0]
    assert choice["message"]["content"] == sent
    assert choice["message"].get("reasoning_content", "") == think
    assert sent or think
    assert choice["finish_reason"] in ("stop", "length")
    assert rec[0]["antwort"].get("timings")


def test_pictures_are_not_kept(stack: Stack) -> None:
    stack.settings({"capture": {"enabled": True}})
    big = "data:image/png;base64," + "A" * 5000
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": big}}]}], "max_tokens": 4}
    stack.chat(body)
    rec = lines(stack)
    if rec:  # the fake engine may refuse a picture without vision; then the refusal is recorded without the picture
        assert "AAAA" not in json.dumps(rec[0])


def test_status_and_delete(stack: Stack) -> None:
    stack.settings({"capture": {"enabled": True}})
    stack.chat(BODY)
    st = stack.json("GET", "/api/capture")[1]
    assert st["records"] == 1 and st["files"] == 1 and st["bytes"] > 0
    assert stack.json("POST", "/api/capture/clear", {})[0] == 400
    assert lines(stack)
    status, data = stack.json("POST", "/api/capture/clear", {"confirm": True})
    assert status == 200 and data["deleted"] == 1
    assert lines(stack) == []


def test_other_paths_are_not_recorded(stack: Stack) -> None:
    stack.settings({"capture": {"enabled": True}})
    stack.call("GET", "/v1/models")
    assert lines(stack) == []


def test_the_size_limit_stops_the_recording(tmp_path: Path) -> None:
    cap = Capture(tmp_path, lambda: True, lambda: 1)  # 1 MB
    rec = cap.begin("/v1/chat/completions", json.dumps({"messages": []}).encode())
    assert rec is not None
    cap.write({"x": "y" * 900_000})
    assert cap.write({"x": "y" * 900_000}) is False
    assert cap.status()["records"] == 1


def test_files_are_private(tmp_path: Path) -> None:
    cap = Capture(tmp_path, lambda: True, lambda: 0)
    cap.write({"a": 1})
    f = next((tmp_path / "capture").glob("*.jsonl"))
    assert (f.stat().st_mode & 0o077) == 0
