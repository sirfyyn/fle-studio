"""The live sampler runs only while somebody looks (a request for /api/live or /api/engine)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest

from fle_studio import metrics


class StubEngine:
    """Only what Live touches; _sample is replaced on the instance in these tests."""

    cache: dict[str, Any] = {}
    accept_logged = None


def wait_for(check: Any, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(0.02)
    return False


class Probe:
    """A Live whose sampling is counted. Not patched on the class: other Studios of the test run have their own
    sampler threads, and those must go on calling the real one."""

    def __init__(self, watch_seconds: float) -> None:
        self.ticks: list[int] = []
        self.live = metrics.Live(StubEngine(), interval=0.05, watch_seconds=watch_seconds)  # type: ignore[arg-type]
        # The new sampler waits for the first look, so nothing runs before this line.
        self.live._sample = lambda tick: self.ticks.append(tick)  # type: ignore[method-assign]


@pytest.fixture
def probe() -> Iterator[Any]:
    made: list[Probe] = []

    def make(watch_seconds: float) -> Probe:
        p = Probe(watch_seconds)
        made.append(p)
        return p

    yield make
    for p in made:  # its thread idles from now on
        p.live.watch_seconds = 0.0


def test_nothing_is_sampled_until_somebody_looks(probe: Any) -> None:
    p = probe(3.0)
    time.sleep(0.5)
    assert p.ticks == [] and not p.live.watched()

    p.live.touch()
    assert p.live.watched()
    assert wait_for(lambda: len(p.ticks) >= 2), "sampling starts at the first look"


def test_sampling_stops_a_little_after_the_last_look(probe: Any) -> None:
    p = probe(0.6)
    p.live.snapshot()  # a page asking for the data is a look
    assert wait_for(lambda: len(p.ticks) >= 1)
    assert wait_for(lambda: not p.live.watched())
    time.sleep(0.6)  # let a sample that was already under way finish
    count = len(p.ticks)
    time.sleep(0.8)
    assert len(p.ticks) == count, "no sample while nobody looks"

    p.live.touch()  # a new page wakes it again
    assert wait_for(lambda: len(p.ticks) > count)
