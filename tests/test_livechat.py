"""Live chat: everything through /v1 as a temporary chat in memory, joined by history, capped, starred."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from chat_support import Stack, make_stack

from fle_studio.livechat import MAX_MESSAGES, LiveChat


def record(messages: list[dict[str, Any]], answer: str, status: int = 200, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "anfrage": {"messages": messages},
        "antwort": {
            "model": "m",
            "choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            **extra,
        },
    }


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def bot(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def texts(thread: dict[str, Any]) -> list[str]:
    return [m["content"] for m in thread["messages"]]


# ------------------------------------------------------------------------------------------------ the store itself


def test_one_request_is_one_thread_with_the_question_and_the_answer() -> None:
    live = LiveChat()
    live.add(record([user("hello there")], "hi"))
    snap = live.snapshot()
    assert snap["count"] == 2 and snap["max_messages"] == MAX_MESSAGES
    (thread,) = snap["threads"]
    assert thread["title"] == "hello there" and thread["model"] == "m"
    assert [(m["role"], m["content"]) for m in thread["messages"]] == [("user", "hello there"), ("assistant", "hi")]


def test_a_request_that_repeats_the_history_joins_the_same_thread() -> None:
    live = LiveChat()
    live.add(record([user("one")], "a"))
    live.add(record([user("one"), bot("a"), user("two")], "b"))
    live.add(record([user("one"), bot("a"), user("two"), bot("b"), user("three")], "c"))
    (thread,) = live.snapshot()["threads"]
    assert texts(thread) == ["one", "a", "two", "b", "three", "c"]


def test_a_different_conversation_is_its_own_thread_and_the_newest_comes_first() -> None:
    live = LiveChat()
    live.add(record([user("first")], "a"))
    live.add(record([user("other")], "b"))
    live.add(record([user("first"), bot("a"), user("again")], "c"))
    first, other = live.snapshot()["threads"]
    assert texts(first) == ["first", "a", "again", "c"] and texts(other) == ["other", "b"]


def test_the_same_question_asked_again_starts_a_new_thread() -> None:
    live = LiveChat()
    live.add(record([user("same")], "a"))
    live.add(record([user("same")], "b"))
    assert len(live.snapshot()["threads"]) == 2


def test_a_system_prompt_belongs_to_the_thread() -> None:
    live = LiveChat()
    sys_msg = {"role": "system", "content": "be brief"}
    live.add(record([sys_msg, user("q")], "a"))
    live.add(record([sys_msg, user("q"), bot("a"), user("q2")], "b"))
    (thread,) = live.snapshot()["threads"]
    assert [m["role"] for m in thread["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert thread["title"] == "q"


def test_a_plain_completion_is_kept_too() -> None:
    live = LiveChat()
    live.add({"status": 200, "anfrage": {"prompt": "once upon"}, "antwort": {"choices": [{"text": " a time"}]}})
    (thread,) = live.snapshot()["threads"]
    assert texts(thread) == ["once upon", " a time"]


def test_failed_and_empty_requests_leave_nothing() -> None:
    live = LiveChat()
    live.add(record([user("x")], "y", status=500))
    live.add({"status": 200, "anfrage": {"messages": []}, "antwort": {"choices": []}})
    live.add({"status": 200, "anfrage": "nonsense", "antwort": None})
    assert live.snapshot()["count"] == 0


def test_pictures_are_left_out_and_timings_kept() -> None:
    live = LiveChat()
    big = "data:image/png;base64," + "A" * 400
    content = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": big}}]
    live.add(record([{"role": "user", "content": content}], "a cat", timings={"predicted_per_second": 81.5}))
    (thread,) = live.snapshot()["threads"]
    assert big not in str(thread)
    assert "image removed" in str(thread["messages"][0]["content"])
    assert thread["messages"][1]["timings"] == {"predicted_per_second": 81.5}


def test_at_most_100_messages_the_oldest_fall_away() -> None:
    live = LiveChat()
    for n in range(60):
        live.add(record([user(f"q{n}")], f"a{n}"))
    snap = live.snapshot()
    assert snap["count"] == MAX_MESSAGES
    kept = [t["messages"][0]["content"] for t in snap["threads"]]
    assert kept[0] == "q59" and "q0" not in kept and "q9" not in kept  # 10 whole threads are gone, 50 remain


def test_one_long_conversation_loses_its_start_and_still_joins() -> None:
    live = LiveChat(max_messages=6)
    history: list[dict[str, Any]] = []
    for n in range(5):
        history.append(user(f"q{n}"))
        live.add(record(list(history), f"a{n}"))
        history.append(bot(f"a{n}"))
    (thread,) = live.snapshot()["threads"]
    assert thread["cut"] is True and len(thread["messages"]) == 6
    assert texts(thread) == ["q2", "a2", "q3", "a3", "q4", "a4"]


def test_delete_and_clear() -> None:
    live = LiveChat()
    live.add(record([user("a")], "x"))
    live.add(record([user("b")], "y"))
    first = live.snapshot()["threads"][0]["id"]
    assert live.delete(first) is True and live.delete(first) is False
    assert len(live.snapshot()["threads"]) == 1
    assert live.clear() == 1 and live.snapshot()["count"] == 0


def test_star_data_is_a_chain_up_to_the_message() -> None:
    live = LiveChat()
    live.add(record([user("one")], "a"))
    live.add(record([user("one"), bot("a"), user("two")], "b"))
    thread = live.snapshot()["threads"][0]
    whole = live.star_data(thread["id"])
    assert [m["content"] for m in whole["messages"]] == ["one", "a", "two", "b"]
    assert [m["parent"] for m in whole["messages"]] == [None] + [m["id"] for m in whole["messages"][:-1]]
    assert whole["leaf"] == whole["messages"][-1]["id"] and whole["favorite"] is True
    part = live.star_data(thread["id"], thread["messages"][1]["id"])
    assert [m["content"] for m in part["messages"]] == ["one", "a"]
    with pytest.raises(KeyError):
        live.star_data(thread["id"], "nope")
    with pytest.raises(KeyError):
        live.star_data("0" * 12)


# ------------------------------------------------------------------------------------------------ through the proxy


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Stack]:
    yield from make_stack(tmp_path)


BODY = {"messages": [{"role": "user", "content": "Say hello in one short sentence."}], "max_tokens": 16}


def live_threads(stack: Stack) -> list[dict[str, Any]]:
    status, snap = stack.json("GET", "/api/livechat")
    assert status == 200
    return snap["threads"]


def test_a_request_through_v1_shows_up_live_without_capture(stack: Stack) -> None:
    assert live_threads(stack) == []
    status, answer = stack.chat(BODY)
    assert status == 200
    (thread,) = live_threads(stack)
    assert [m["role"] for m in thread["messages"]] == ["user", "assistant"]
    assert thread["messages"][0]["content"] == BODY["messages"][0]["content"]
    assert thread["messages"][1]["content"] == answer["choices"][0]["message"]["content"]
    assert not (stack.studio.data_dir / "capture").exists()  # memory only


def test_a_streamed_request_and_its_follow_up_make_one_thread(stack: Stack) -> None:
    status, _, _ = stack.call("POST", "/v1/chat/completions", {**BODY, "stream": True})
    assert status == 200
    (first,) = live_threads(stack)
    answer = first["messages"][1]["content"]
    follow = {"messages": [*BODY["messages"], {"role": "assistant", "content": answer}, user("And again?")]}
    assert stack.chat(follow)[0] == 200
    (thread,) = live_threads(stack)
    assert len(thread["messages"]) == 4 and thread["messages"][2]["content"] == "And again?"


def test_nothing_goes_to_the_chat_archive_until_it_is_starred(stack: Stack) -> None:
    stack.chat(BODY)
    assert stack.json("GET", "/api/chats")[1]["chats"] == []
    (thread,) = live_threads(stack)
    status, saved = stack.json("POST", f"/api/livechat/{thread['id']}/star", {})
    assert status == 200 and saved["favorite"] is True
    chats = stack.json("GET", "/api/chats")[1]["chats"]
    assert len(chats) == 1 and chats[0]["favorite"] is True and chats[0]["messages"] == 2
    full = stack.json("GET", f"/api/chats/{saved['id']}")[1]
    assert [m["role"] for m in full["messages"]] == ["user", "assistant"]
    assert live_threads(stack)[0]["starred"] is True


def test_star_one_message_keeps_the_conversation_up_to_it(stack: Stack) -> None:
    stack.chat(BODY)
    (thread,) = live_threads(stack)
    status, saved = stack.json("POST", f"/api/livechat/{thread['id']}/star", {"message": thread["messages"][0]["id"]})
    assert status == 200
    assert [m["role"] for m in stack.json("GET", f"/api/chats/{saved['id']}")[1]["messages"]] == ["user"]


def test_capture_is_off_by_default_and_live_works_anyway(stack: Stack) -> None:
    assert stack.studio.config.get("capture.enabled") is False
    stack.chat(BODY)
    assert len(live_threads(stack)) == 1
    assert not (stack.studio.data_dir / "capture").exists()


def test_delete_and_clear_over_the_api(stack: Stack) -> None:
    stack.chat(BODY)
    stack.chat({"messages": [user("something else")]})
    first, second = live_threads(stack)
    assert stack.json("DELETE", f"/api/livechat/{first['id']}")[0] == 200
    assert stack.json("DELETE", f"/api/livechat/{first['id']}")[0] == 404
    assert [t["id"] for t in live_threads(stack)] == [second["id"]]
    assert stack.json("DELETE", "/api/livechat")[1]["deleted"] == 1
    assert live_threads(stack) == []
    assert stack.json("POST", "/api/livechat/000000000000/star", {})[0] == 404
