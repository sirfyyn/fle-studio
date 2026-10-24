"""If the engine dies while loading, the automatic profile starts upstream llama.cpp once instead.

Real processes: the stand-in server plays both the FLE engine (a copy of the file, so the two paths differ; it fails the
load like a CUDA out-of-memory error) and the installed upstream llama-server.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from helpers import FAKE_SERVER, stub_gguf, wait_for

from fle_studio.launch import Plan, free_port
from fle_studio.server import Studio


@pytest.fixture
def studio(tmp_path: Path) -> Iterator[Studio]:
    st = Studio(tmp_path / "home")
    original = st.engine.on_load_failure
    st.engine.hook_calls = []  # type: ignore[attr-defined]  # what the engine asked and what the Studio answered

    def recording(failed: Plan, rc: int, error: str) -> Plan | None:
        answer = original(failed, rc, error) if original else None
        st.engine.hook_calls.append(answer)  # type: ignore[attr-defined]
        return answer

    st.engine.on_load_failure = recording
    yield st
    st.engine.shutdown()


@pytest.fixture
def parts(tmp_path: Path, studio: Studio, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """(a model, the 'FLE' binary, the 'upstream' engine record) with upstream installed."""
    model = stub_gguf(tmp_path / "m.gguf")
    fle_binary = tmp_path / "fle" / "llama-server.py"
    fle_binary.parent.mkdir()
    shutil.copy(FAKE_SERVER, fle_binary)
    upstream = {
        "kind": "upstream",
        "flavor": "upstream",
        "backend": "cpu",
        "path": str(FAKE_SERVER),
        "log_verbosity": True,
        "spec_type": True,
    }
    monkeypatch.setattr(studio, "_upstream_engine", lambda info: upstream)
    return {"model": model, "fle": fle_binary, "upstream": upstream}


def failing_plan(parts: dict[str, Any], profile: str = "auto", env: dict[str, str] | None = None) -> Plan:
    port = free_port()
    model = {"id": "m", "name": "stub", "path": str(parts["model"])}
    args = ["-m", str(parts["model"]), "--host", "127.0.0.1", "--port", str(port), "--spec-type", "draft-mtp"]
    return Plan(
        engine="fle",
        binary=str(parts["fle"]),
        args=args,
        env={"FAKE_ENGINE": "fle", "FAKE_FAIL": "oom", "FAKE_LOAD_SECONDS": "0.1", **(env or {})},
        port=port,
        model=model,
        profile=profile,
    )


def starts(studio: Studio) -> int:
    return sum(1 for _, _, text in studio.engine.lines if text.startswith("$ "))


def stays_crashed(studio: Studio, starts_expected: int = 1, asked: int = 1) -> None:
    """The engine ends crashed after exactly ``starts_expected`` starts, and was asked about a fallback ``asked`` times.

    A crash is also the state between the failed load and the fallback start, so the end is told by the start count
    and by the Studio's answer, not by a pause."""
    if asked:
        wait_for(lambda: len(studio.engine.hook_calls) >= asked, 40, "the Studio to be asked")
    wait_for(lambda: studio.engine.state == "crashed" and starts(studio) == starts_expected, 40, "the final crash")
    time.sleep(1.0)  # nothing may follow
    assert studio.engine.state == "crashed" and starts(studio) == starts_expected
    assert len(studio.engine.hook_calls) == asked


def test_a_failed_load_starts_upstream_once(studio: Studio, parts: dict[str, Any]) -> None:
    studio.engine.start(failing_plan(parts))
    wait_for(lambda: studio.engine.state == "ready", 40, "the fallback engine to be ready")
    plan = studio.engine.plan
    assert plan and plan.engine == "upstream" and plan.binary == str(FAKE_SERVER)
    assert "--spec-type" not in plan.args and "-md" not in plan.args
    assert "ended while loading" in plan.fallback and plan.notes[0] == plan.fallback
    text = [t for _, _, t in studio.engine.lines]
    assert any(t.startswith("FLE Studio: The engine ended while loading") for t in text)
    assert starts(studio) == 2
    assert studio.engine.status()["plan"]["fallback"] == plan.fallback


def test_the_fallback_is_not_tried_a_second_time(studio: Studio, parts: dict[str, Any]) -> None:
    studio.config.update({"launch": {"extra_env": "FAKE_FAIL=oom"}})  # the upstream plan fails to load as well
    studio.engine.start(failing_plan(parts))
    stays_crashed(studio, starts_expected=2)
    assert studio.engine.plan and studio.engine.plan.fallback


def test_switched_off_there_is_no_fallback(studio: Studio, parts: dict[str, Any]) -> None:
    studio.config.update({"launch": {"fallback_on_failure": False}})
    studio.engine.start(failing_plan(parts))
    stays_crashed(studio)


def test_a_profile_or_a_missing_upstream_engine_gets_no_fallback(
    studio: Studio, parts: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    studio.engine.start(failing_plan(parts, profile="my-own-command"))
    stays_crashed(studio)
    studio.engine.stop()
    monkeypatch.setattr(studio, "_upstream_engine", lambda info: None)
    studio.engine.lines.clear()
    studio.engine.start(failing_plan(parts))
    stays_crashed(studio, asked=2)


def test_a_crash_after_ready_is_not_a_failed_load(studio: Studio, parts: dict[str, Any]) -> None:
    plan = failing_plan(parts)
    plan.env.pop("FAKE_FAIL")
    plan.env["FAKE_CRASH_AFTER"] = "0.5"
    studio.engine.start(plan)
    wait_for(lambda: studio.engine.state == "ready", 20, "the engine to be ready")
    stays_crashed(studio, asked=0)


def test_stopping_while_loading_is_not_a_failure(studio: Studio, parts: dict[str, Any]) -> None:
    plan = failing_plan(parts)
    plan.env.pop("FAKE_FAIL")
    plan.env["FAKE_LOAD_SECONDS"] = "30"
    studio.engine.start(plan)
    wait_for(lambda: studio.engine.proc is not None, 10, "the process")
    studio.engine.stop()
    wait_for(lambda: studio.engine.state == "stopped", 30, "the engine to stop")
    time.sleep(1.0)
    assert studio.engine.state == "stopped" and starts(studio) == 1
