"""The live sampler runs only while somebody looks (a request for /api/live or /api/engine)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from fle_studio import metrics


class StubEngine:
    """Only what Live touches; _sample is replaced in these tests."""

    cache: dict[str, Any] = {}
    accept_logged = None


def wait_for(check: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def samples(monkeypatch: pytest.MonkeyPatch) -> Any:
    seen: list[int] = []
    made: list[metrics.Live] = []
    monkeypatch.setattr(metrics.Live, "_sample", lambda self, tick: seen.append(tick))
    original = metrics.Live.__init__

    def init(self: metrics.Live, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(metrics.Live, "__init__", init)
    yield seen
    for live in made:  # its thread idles from now on and must not count samples of the next test
        live.watch_seconds = 0.0
    time.sleep(0.4)


def test_nothing_is_sampled_until_somebody_looks(samples: list[int]) -> None:
    live = metrics.Live(StubEngine(), interval=0.05, watch_seconds=3.0)  # type: ignore[arg-type]
    time.sleep(0.5)
    assert samples == [] and not live.watched()

    live.touch()
    assert live.watched()
    assert wait_for(lambda: len(samples) >= 2), "sampling starts at the first look"


def test_sampling_stops_a_little_after_the_last_look(samples: list[int]) -> None:
    live = metrics.Live(StubEngine(), interval=0.05, watch_seconds=0.6)  # type: ignore[arg-type]
    live.snapshot()  # a page asking for the data is a look
    assert wait_for(lambda: len(samples) >= 1)
    assert wait_for(lambda: not live.watched())
    time.sleep(0.6)  # let a sample that was already under way finish
    count = len(samples)
    time.sleep(0.8)
    assert len(samples) == count, "no sample while nobody looks"

    live.touch()  # a new page wakes it again
    assert wait_for(lambda: len(samples) > count)
