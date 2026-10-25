"""The change of the decode speed against the last speed test of the same model and settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helpers import fake_plan, start_engine, stop_engine, stub_gguf, wait_for

from fle_studio.bench import (
    Bench,
    compare_decode,
    compare_with_history,
    decode_series,
    setting_key,
)
from fle_studio.engine import Engine

CTX = {"model": "M", "n_ctx": 32768, "command": "FOO=1 \\\n  llama-server -m m.gguf --port 8081 -c 32768"}


def one_request(name: str, decode: float, *, truncated: bool | None = False) -> dict[str, Any]:
    t: dict[str, Any] = {"name": name, "decode": decode, "run": 1}
    if truncated is not None:
        t["truncated"] = truncated
    return t


def record(
    ident: str, decodes: dict[str, float], *, ctx: dict[str, Any] | None = None, **flags: bool
) -> dict[str, Any]:
    tests = [one_request(n, v, truncated=flags.get(n, False)) for n, v in decodes.items()]
    return {"id": ident, "state": "done", "started": 1.0, "context": ctx or CTX, "tests": tests}


def test_a_drop_of_more_than_ten_percent_warns() -> None:
    old = record("a", {"Short answer": 100.0, "Code": 100.0})
    new = record("b", {"Short answer": 85.0, "Code": 89.0})
    c = compare_decode(new, old)
    assert c["comparable"] and c["old"] == 100.0 and c["new"] == 87.0
    assert c["change_pct"] == -13.0 and c["warn"] is True


def test_ten_percent_exactly_and_gains_do_not_warn() -> None:
    old = record("a", {"Short answer": 100.0})
    assert compare_decode(record("b", {"Short answer": 90.0}), old)["warn"] is False
    c = compare_decode(record("b", {"Short answer": 120.0}), old)
    assert c["change_pct"] == 20.0 and c["warn"] is False


def test_truncated_requests_are_left_out() -> None:
    old = record("a", {"Short answer": 100.0, "Code": 50.0})
    new = record("b", {"Short answer": 80.0, "Code": 10.0}, Code=True)  # Code hit the token limit in the new run
    c = compare_decode(new, old)
    assert c["comparable"] and c["tests"] == ["Short answer"] and c["old"] == 100.0 and c["change_pct"] == -20.0


def test_no_comparison_when_everything_was_cut_off() -> None:
    old = record("a", {"Short answer": 100.0, "Code": 50.0}, **{"Short answer": True, "Code": True})
    new = record("b", {"Short answer": 80.0, "Code": 40.0})
    c = compare_decode(new, old)
    assert c["comparable"] is False and "token limit" in c["reason"] and "change_pct" not in c


def test_results_without_the_truncated_field_are_not_compared() -> None:
    old = {
        "id": "a",
        "tests": [one_request("Short answer", 100.0, truncated=None)],
    }  # measured before the fixed sampling
    assert compare_decode(record("b", {"Short answer": 50.0}), old)["comparable"] is False


def test_the_prefill_request_is_not_part_of_the_decode_comparison() -> None:
    old = record("a", {"Long prompt": 100.0})
    assert compare_decode(record("b", {"Long prompt": 10.0}), old)["comparable"] is False


def test_the_port_does_not_make_a_different_setting() -> None:
    other = {**CTX, "command": CTX["command"].replace("--port 8081", "--port=9999")}
    assert setting_key(CTX) == setting_key(other)
    assert setting_key(CTX) != setting_key({**CTX, "command": CTX["command"] + " --flash-attn on"})
    assert setting_key(CTX) != setting_key({**CTX, "model": "N"})
    assert setting_key(CTX) != setting_key({**CTX, "n_ctx": 65536})


def test_without_a_command_the_profile_and_engine_decide() -> None:
    a = {"model": "M", "n_ctx": 1, "command": None, "profile": "service x", "engine": "", "build": "b1"}
    assert setting_key(a) == setting_key(dict(a))
    assert setting_key(a) != setting_key({**a, "build": "b2"})


def test_history_picks_the_newest_matching_finished_result() -> None:
    new = record("n", {"Short answer": 90.0})
    history = [  # newest first
        record("other-setting", {"Short answer": 1.0}, ctx={**CTX, "model": "N"}),
        {**record("cancelled", {"Short answer": 1.0}), "state": "cancelled"},
        record("n", {"Short answer": 1.0}),  # the result itself, if it were saved already
        record("match", {"Short answer": 100.0}),
        record("older", {"Short answer": 500.0}),
    ]
    c = compare_with_history(new, history)
    assert c and c["previous"] == "match" and c["change_pct"] == -10.0
    assert compare_with_history(new, history[:3]) is None


def test_the_series_is_the_same_setting_oldest_first_ending_with_the_new_run() -> None:
    new = record("n", {"Short answer": 90.0, "Code": 110.0})
    history = [  # newest first
        record("other-setting", {"Short answer": 1.0}, ctx={**CTX, "model": "N"}),
        {**record("cancelled", {"Short answer": 1.0}), "state": "cancelled"},
        record("n", {"Short answer": 1.0}),  # the result itself, if it were saved already
        record("c", {"Short answer": 100.0}),
        record("b", {"Short answer": 80.0, "Code": 60.0}),
        record("a", {"Short answer": 70.0}),
    ]
    series = decode_series(new, history)
    assert [p["id"] for p in series] == ["a", "b", "c", "n"]
    assert [p["decode"] for p in series] == [70.0, 70.0, 100.0, 100.0]  # medians of the requests per run
    assert compare_with_history(new, history)["series"] == series  # type: ignore[index]


def test_runs_cut_off_at_the_token_limit_or_without_the_field_have_no_value() -> None:
    cut = record("cut", {"Short answer": 5.0, "Code": 5.0}, **{"Short answer": True, "Code": True})
    old = {
        "id": "old",
        "state": "done",
        "started": 0.5,
        "context": CTX,
        "tests": [one_request("Code", 9, truncated=None)],
    }
    series = decode_series(record("n", {"Short answer": 90.0}), [cut, old])
    assert [(p["id"], p["decode"]) for p in series] == [("old", None), ("cut", None), ("n", 90.0)]


def test_the_series_is_capped_and_empty_history_is_just_the_run() -> None:
    history = [record(f"r{i}", {"Short answer": float(i + 1)}) for i in range(20)]
    series = decode_series(record("n", {"Short answer": 1.0}), history, limit=5)
    assert len(series) == 5 and series[-1]["id"] == "n" and series[0]["id"] == "r3"
    assert [p["id"] for p in decode_series(record("n", {"Short answer": 1.0}), [])] == ["n"]


def test_a_speed_test_stores_the_comparison_with_the_one_before(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("FAKE_LOAD_SECONDS", "0.1")
    engine = Engine(tmp_path / "data")
    plan = fake_plan(stub_gguf(tmp_path / "m.gguf"))
    try:
        assert start_engine(engine, plan) == "ready"
        bench = Bench(engine, tmp_path / "data")
        ctx = {"model": "stub", "n_ctx": 4096, "command": plan.command_line()}
        first = bench.start(runs=1, label="one", context=ctx)
        wait_for(lambda: (bench.status() or {}).get("state") in ("done", "failed"), 120, "the first speed test")
        assert bench.status()["state"] == "done" and "compare" not in bench.status()
        bench.start(runs=1, label="two", context=dict(ctx))
        wait_for(
            lambda: (bench.status() or {}).get("id") != first["id"] and bench.status()["state"] in ("done", "failed"),
            120,
            "the second speed test",
        )
        status = bench.status()
        assert status["state"] == "done"
        c = status["compare"]
        assert c["previous"] == first["id"]
        assert c["comparable"] is False or isinstance(c["change_pct"], float)
        saved = [r for r in bench.results() if r["id"] == status["id"]]
        assert saved and saved[0]["compare"] == c  # kept in the stored result too
    finally:
        stop_engine(engine)
