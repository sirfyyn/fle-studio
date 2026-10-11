"""The expert-cache view: parsing of the engine's cache lines, the series built from them, and the real thing against
the stand-in engine."""

from __future__ import annotations

from pathlib import Path

from helpers import chat, fake_plan, start_engine, stop_engine, stub_gguf, wait_for

from fle_studio.cachewatch import CacheWatch
from fle_studio.engine import Engine

ENABLED = (
    "0.01.234.567 I operator(): MoE expert cache enabled: 4 layers x 10 slots, 1 inserts/step, 156.0 MiB device memory"
)


def feed_round(watch: CacheWatch, steps: int, hits: int, misses: int, t: float, with_layers: bool = True) -> None:
    rate = 100 * hits / (hits + misses)
    watch.feed(f"x I moe-cache: steps={steps} hits={hits} misses={misses} hit-rate={rate:.1f}%", t)
    if not with_layers:
        return
    for layer in range(4):
        h, m = hits // 4, misses // 4
        watch.feed(
            f"moe-layer: steps={steps} layer={layer} hits={h} misses={m} "
            f"filled={min(10, steps // 100 + layer)} slots=10",
            t,
        )
        watch.feed(f"moe-hot: steps={steps} layer={layer} top={layer}:50,{layer + 10}:40,{layer + 20}:30", t)


def test_overall_lines_alone_give_the_overall_view_and_say_what_is_missing() -> None:
    w = CacheWatch()
    w.feed(ENABLED)
    feed_round(w, 512, 700, 300, 1.0, with_layers=False)
    feed_round(w, 1024, 1700, 300, 2.0, with_layers=False)
    snap = w.snapshot()
    assert snap["available"] == {"overall": True, "layers": False, "hot": False}
    assert snap["enabled"]["layers"] == 4 and snap["enabled"]["slots"] == 10
    assert [p["steps"] for p in snap["overall"]["series"]] == [512, 1024]
    assert snap["overall"]["hit_rate"] == 85.0
    assert snap["overall"]["recent_rate"] == 100.0  # 1000 hits, 0 misses between the two lines
    assert snap["layers"] == [] and snap["fill"]["series"] == []
    assert len(snap["missing"]) == 2 and "moe-layer" in snap["missing"][0] and "moe-hot" in snap["missing"][1]


def test_layer_and_hot_lines_fill_the_per_layer_view() -> None:
    w = CacheWatch()
    w.feed(ENABLED)
    feed_round(w, 512, 800, 200, 1.0)
    feed_round(w, 1024, 1900, 300, 2.0)
    snap = w.snapshot(layer=2)
    assert snap["available"] == {"overall": True, "layers": True, "hot": True}
    assert snap["missing"] == []
    assert [x["layer"] for x in snap["layers"]] == [0, 1, 2, 3]
    layer = snap["layers"][2]
    assert layer["hits"] == 475 and layer["misses"] == 75 and layer["hit_rate"] == round(100 * 475 / 550, 2)
    assert layer["slots"] == 10 and layer["filled"] == 10 and layer["fill"] == 1.0
    assert layer["recent_rate"] == round(100 * (475 - 200) / (475 - 200 + 75 - 50), 2)
    assert snap["hot"]["layers"]["2"] == [(2, 50), (12, 40), (22, 30)]
    assert len(snap["layer_series"]["series"]) == 2
    assert [p["steps"] for p in snap["fill"]["series"]] == [512, 1024]
    assert 0 < snap["fill"]["now"] <= 1


def test_a_new_run_starts_from_zero() -> None:
    w = CacheWatch()
    w.feed(ENABLED)
    feed_round(w, 1024, 800, 200, 1.0)
    w.feed("$ /engine/llama-server -m model.gguf")  # FLE Studio logs this when it starts an engine
    assert w.snapshot()["available"] == {"overall": False, "layers": False, "hot": False}
    feed_round(w, 512, 800, 200, 2.0)
    w.feed(ENABLED)  # a service that restarted prints it again
    assert w.snapshot()["available"]["overall"] is False


def test_steps_going_back_start_a_new_series() -> None:
    w = CacheWatch()
    feed_round(w, 1024, 800, 200, 1.0, with_layers=False)
    feed_round(w, 512, 500, 500, 2.0, with_layers=False)
    assert [p["steps"] for p in w.snapshot()["overall"]["series"]] == [512]


def test_other_lines_are_not_cache_lines() -> None:
    w = CacheWatch()
    assert not w.feed("srv  llama_server: listening on http://127.0.0.1:8080")
    assert not w.feed("moe-hot: not a number")
    assert w.snapshot()["lines"] == 0


def test_series_stay_short() -> None:
    w = CacheWatch()
    for i in range(1, 1000):
        feed_round(w, 512 * i, 700 * i, 300 * i, float(i), with_layers=False)
    assert len(w.snapshot()["overall"]["series"]) == 400


def test_the_stand_in_engine_feeds_the_view_through_the_engine_log(tmp_path: Path) -> None:
    model = stub_gguf(tmp_path / "m" / "stub-Q4_K_M.gguf")
    engine = Engine(tmp_path / "data")
    watch = CacheWatch()
    engine.line_listeners.append(watch.feed)
    plan = fake_plan(
        model,
        env={
            "FLE_MOE_CACHE_BUDGET_MIB": "2000",
            "FAKE_ANSWER_TOKENS": "700",
            "FAKE_TOK_PER_SEC": "20000",
            "FAKE_THINK_TOKENS": "0",
        },
        engine="fle",
    )
    try:
        assert start_engine(engine, plan) == "ready"
        chat(engine.base_url, "hello", max_tokens=1000)
        snap = wait_for(lambda: (s := watch.snapshot(layer=0))["available"]["hot"] and s, 20, "the cache lines")
    finally:
        stop_engine(engine)
    assert snap["available"] == {"overall": True, "layers": True, "hot": True}
    assert snap["enabled"]["layers"] == 6  # the stub model has 6 layers
    assert len(snap["layers"]) == 6
    first = snap["layers"][0]
    assert first["slots"] and first["filled"] is not None and first["hit_rate"] is not None
    assert len(snap["hot"]["layers"]["0"]) == 16
    # the aggregate line is the sum of the layer lines (up to the rounding of cumulative counts)
    total_hits = sum(x["hits"] for x in snap["layers"])
    assert abs(total_hits - snap["overall"]["series"][-1]["hits"]) <= 6
