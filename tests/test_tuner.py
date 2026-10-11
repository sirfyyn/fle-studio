"""The tuner and the A/B/A comparison against the stand-in engine (FAKE_SPEED_MODEL=1: a sweet spot per setting)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from helpers import fake_plan, stub_gguf, wait_for

from fle_studio import bench, trials, tuner
from fle_studio.engine import Engine
from fle_studio.launch import Plan, Profile, Profiles

SYSINFO = {"cpu": {"name": "Test CPU", "physical": 7}, "memory": {"total": 64 * 2**30}, "gpus": []}
FAST = {
    "FAKE_SPEED_MODEL": "1",
    "FAKE_TOK_PER_SEC": "3000",
    "FAKE_PROMPT_TOK_PER_SEC": "200000",
    "FAKE_JITTER": "0",
    "FAKE_RUN_NOISE": "0.01",
}


class Setup:
    def __init__(self, tmp: Path, env: dict[str, str] | None = None) -> None:
        self.model = stub_gguf(tmp / "m.gguf")
        self.engine = Engine(tmp)
        self.profiles = Profiles(tmp)
        self.env = {**FAST, **(env or {})}
        self.tuner = tuner.Tuner(
            tuner.Host(
                engine=self.engine,
                bench=bench.Bench(self.engine, tmp),
                profiles=self.profiles,
                data_dir=tmp,
                make_plan=self.make_plan,
                sysinfo=lambda: SYSINFO,
                ready_timeout=30,
            )
        )

    def make_plan(self, model_id: str, profile: str | None) -> Plan:
        prof = self.profiles.get(profile) if profile and profile != "auto" else None
        extra = list(prof.args) if prof else []
        env = {**self.env, **(prof.env if prof else {"FLE_MOE_CACHE_BUDGET_MIB": "10000"})}
        return fake_plan(self.model, *[a for a in extra if a != "{model}"], env=env)

    def finished(self, timeout: float = 120) -> dict[str, Any]:
        cur = self.tuner.current
        assert cur is not None
        wait_for(lambda: cur["state"] != "running", timeout, "the run to finish")
        return cur


@pytest.fixture
def setup(tmp_path: Path):
    s = Setup(tmp_path)
    yield s
    s.tuner.cancel()
    s.engine.stop()


def test_set_flag_replaces_every_spelling_and_adds_missing() -> None:
    assert trials.set_flag(["-m", "x", "--threads", "4"], "-t", "6") == ["-m", "x", "--threads", "6"]
    assert trials.set_flag(["-m", "x"], "-b", "1024") == ["-m", "x", "-b", "1024"]
    assert trials.set_flag(["-t", "4", "--jinja"], "-t", None) == ["--jinja"]
    assert trials.set_flag(["--n-cpu-moe=4"], "--n-cpu-moe", "8") == ["--n-cpu-moe=8"]


def test_candidates_depend_on_the_engine_and_the_machine() -> None:
    fle = Plan("fle", "x", ["-t", "6"], {tuner.CACHE_ENV: "10000"}, 1, {"info": {}})
    keys = [p.key for p in tuner.candidates(fle, SYSINFO)]
    assert keys == ["threads", "cache", "batch"]
    up = Plan("upstream", "x", [], {}, 1, {"info": {"moe": True, "layers": 40}})
    gpu = {**SYSINFO, "gpus": [{"vendor": "nvidia", "name": "G", "memory_total": 2**34}]}
    assert [p.key for p in tuner.candidates(up, gpu)] == ["threads", "ncmoe", "batch"]
    assert "ncmoe" not in [p.key for p in tuner.candidates(up, SYSINFO)]  # no GPU, nothing to offload to


def test_the_tuner_finds_the_best_thread_count_and_saves_a_profile(setup: Setup) -> None:
    setup.tuner.start_tune(model_id="m", budget_s=600, runs=3, params=("threads",), screen=False)
    cur = setup.finished()
    assert cur["state"] == "done", cur.get("error")
    assert cur["probe"]["ok"] and cur["baseline"]["decode_runs"]
    assert any(t["state"] == "ok" for t in cur["trials"])
    prof = setup.profiles.get("tuned-m")
    assert prof is not None and prof.name == "Tuned: stub"
    assert prof.args[prof.args.index("-t") + 1] == "6"
    tuned = prof.meta["tuned"]
    assert tuned["machine"] == trials.machine_key(SYSINFO) and tuned["model_id"] == "m"
    assert setup.tuner.tuned_profile("m")["id"] == "tuned-m"
    assert "{model}" in prof.args and "--port" not in prof.args
    assert setup.tuner.results()[0]["id"] == cur["id"]
    assert setup.engine.state in ("stopped", "crashed")


def test_a_profile_from_another_machine_is_not_offered(setup: Setup) -> None:
    setup.profiles.save(
        {
            "id": "tuned-m",
            "name": "Tuned",
            "engine": "fle",
            "args": ["-m", "{model}"],
            "meta": {"tuned": {"model_id": "m", "machine": "other"}},
        }
    )
    assert setup.tuner.tuned_profile("m") is None


def test_a_probe_that_fails_stops_the_run_and_saves_nothing(tmp_path: Path) -> None:
    s = Setup(tmp_path, {"FAKE_FAIL": "oom"})
    try:
        s.tuner.start_tune(model_id="m", budget_s=600, runs=2, params=("threads",), screen=False)
        cur = s.finished()
        assert cur["state"] == "failed" and "nothing was tried" in cur["error"]
        assert cur["probe"]["ok"] is False and cur["trials"] == []
        assert s.profiles.get("tuned-m") is None
    finally:
        s.engine.stop()


def test_a_candidate_that_does_not_load_is_recorded_and_skipped(tmp_path: Path) -> None:
    s = Setup(tmp_path, {"FAKE_CACHE_LIMIT_MIB": "8000"})  # 10000 loads (limit 1.3 x 8000), 10752 does not
    try:
        s.tuner.start_tune(model_id="m", budget_s=600, runs=2, params=("cache",), screen=False)
        cur = s.finished()
        assert cur["state"] == "done", cur.get("error")
        states = {t["label"]: t["state"] for t in cur["trials"]}
        assert states["expert cache 10752 MiB"] == "failed"
        assert states["expert cache 7936 MiB"] == "ok"
    finally:
        s.engine.stop()


def test_the_budget_stops_the_run_before_a_trial_that_would_not_fit(setup: Setup) -> None:
    setup.tuner.start_tune(model_id="m", budget_s=60, runs=2, params=("threads", "cache", "batch"), screen=False)
    cur = setup.tuner.current
    assert cur is not None
    cur["budget_s"] = 1  # shrink it while the probe runs: no further trial fits
    cur = setup.finished()
    assert cur["state"] == "done"
    assert any("time budget" in n for n in cur["notes"])
    assert cur["trials"] == []


def test_cancel_stops_the_engine_and_ends_the_run(setup: Setup) -> None:
    setup.tuner.start_tune(model_id="m", budget_s=600, runs=3, params=("threads", "cache"), screen=False)
    wait_for(lambda: setup.engine.state in ("starting", "ready"), 30, "the engine")
    setup.tuner.cancel()
    cur = setup.finished()
    assert cur["state"] == "cancelled"
    wait_for(lambda: setup.engine.state in ("stopped", "crashed"), 30, "the engine to stop")


def test_a_second_run_is_refused_and_so_is_a_running_engine(setup: Setup) -> None:
    setup.tuner.start_tune(model_id="m", budget_s=600, runs=2, params=("threads",), screen=False)
    with pytest.raises(RuntimeError, match="already running"):
        setup.tuner.start_tune(model_id="m")
    setup.tuner.cancel()
    setup.finished()


def test_aba_tells_a_real_difference_from_noise(setup: Setup) -> None:
    good = setup.profiles.save(
        {"id": "six", "name": "Six threads", "engine": "fle", "args": ["-t", "6"], "env": {tuner.CACHE_ENV: "10000"}}
    )
    bad = setup.profiles.save(
        {"id": "two", "name": "Two threads", "engine": "fle", "args": ["-t", "2"], "env": {tuner.CACHE_ENV: "10000"}}
    )
    assert isinstance(good, Profile) and isinstance(bad, Profile)
    setup.tuner.start_aba(model_id="m", profile_a="six", profile_b="two", runs=3)
    cur = setup.finished()
    assert cur["state"] == "done", cur.get("error")
    assert [t["phase"] for t in cur["trials"]] == ["A1", "B", "A2"]
    assert cur["result"]["decode"]["verdict"] == "worse"
    wait_for(lambda: setup.engine.state in ("stopped", "crashed"), 30, "the engine to stop")
    setup.tuner.start_aba(model_id="m", profile_a="six", profile_b="six", runs=3)
    same = setup.finished()
    assert same["state"] == "done", same.get("error")
    assert same["result"]["decode"]["verdict"] == "within noise"


def test_screen_helpers_read_llama_bench_json() -> None:
    text = (
        'noise\n[{"n_threads": 4, "n_prompt": 256, "n_gen": 0, "samples_ts": [10, 11]}, '
        '{"n_threads": 4, "n_prompt": 0, "n_gen": 32, "samples_ts": [5, 6]}]'
    )
    table = tuner.screen_values(tuner.parse_bench_json(text), "n_threads")
    assert table == {4: {"prefill": [10, 11], "decode": [5, 6]}}
    with pytest.raises(ValueError):
        tuner.parse_bench_json("nothing")
