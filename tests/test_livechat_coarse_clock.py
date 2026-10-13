"""The live chat must keep its order when the clock is coarse (Windows ticks about every 15 ms)."""

from fle_studio import livechat
from fle_studio.livechat import MAX_MESSAGES, LiveChat


def record(question: str, answer: str) -> dict:
    return {
        "status": 200,
        "anfrage": {"messages": [{"role": "user", "content": question}]},
        "antwort": {"choices": [{"message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}]},
    }


def test_order_and_trimming_hold_when_every_request_has_the_same_time(monkeypatch) -> None:
    monkeypatch.setattr(livechat.time, "time", lambda: 1000.0)
    live = LiveChat()
    for n in range(60):
        live.add(record(f"q{n}", f"a{n}"))
    snap = live.snapshot()
    assert snap["count"] == MAX_MESSAGES
    kept = [t["messages"][0]["content"] for t in snap["threads"]]
    assert kept[0] == "q59" and "q0" not in kept and "q9" not in kept
