"""Statistics behind the repeated runs, the 95 % interval and the "within noise" verdict."""

from __future__ import annotations

import math
import random

import pytest

from fle_studio import stats


@pytest.mark.parametrize(
    ("df", "want"),
    [(1, 12.7062), (2, 4.30265), (5, 2.57058), (10, 2.22814), (30, 2.04227), (1000, 1.96234)],
)
def test_t_quantile_matches_the_tables(df: int, want: float) -> None:
    assert stats.t_quantile(0.975, df) == pytest.approx(want, abs=1e-3)
    assert stats.t_quantile(0.025, df) == pytest.approx(-want, abs=1e-3)


def test_t_cdf_is_symmetric_and_monotone() -> None:
    assert stats.t_cdf(0.0, 7) == pytest.approx(0.5)
    assert stats.t_cdf(1.3, 7) + stats.t_cdf(-1.3, 7) == pytest.approx(1.0)
    assert stats.t_cdf(2.0, 7) > stats.t_cdf(1.0, 7)


def noisy(center: float, spread: float, n: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [center + rng.uniform(-spread, spread) for _ in range(n)]


def test_summarize_needs_three_runs_for_an_interval() -> None:
    assert stats.summarize([])["median"] is None
    one = stats.summarize([50.0])
    assert one["median"] == 50.0 and one["lo"] is None
    two = stats.summarize([50.0, 52.0])
    assert two["median"] == 51.0 and two["lo"] is None and two["hi"] is None
    three = stats.summarize([50.0, 52.0, 51.0])
    assert three["method"] == "t"
    assert three["lo"] < 51.0 < three["hi"]


def test_summarize_interval_contains_the_median_and_switches_to_bootstrap() -> None:
    few = stats.summarize(noisy(80, 4, 5, 1))
    many = stats.summarize(noisy(80, 4, 20, 1))
    assert few["method"] == "t" and many["method"] == "bootstrap"
    for s in (few, many):
        assert s["lo"] <= s["median"] <= s["hi"]
    assert (many["hi"] - many["lo"]) < (few["hi"] - few["lo"])  # more runs, tighter interval


def test_summarize_skips_missing_values() -> None:
    assert stats.summarize([None, 10.0, float("nan"), 12.0, 11.0])["n"] == 3


def test_a_real_difference_is_found() -> None:
    a = noisy(80, 2, 5, 2)
    b = noisy(88, 2, 5, 3)
    res = stats.compare(a, b)
    assert res["verdict"] == "better" and res["within_noise"] is False
    assert res["lo"] > 0 and res["method"] == "welch"
    assert res["pct"] == pytest.approx(10, abs=3)
    assert stats.compare(b, a)["verdict"] == "worse"


def test_lower_is_better_flips_the_wording() -> None:
    a, b = noisy(100, 2, 6, 4), noisy(120, 2, 6, 5)  # latency went up
    assert stats.compare(a, b, higher_is_better=False)["verdict"] == "worse"


def test_no_difference_is_within_noise() -> None:
    a = noisy(80, 6, 4, 6)
    b = noisy(80.5, 6, 4, 7)
    res = stats.compare(a, b)
    assert res["verdict"] == "within noise" and res["within_noise"] is True
    assert res["lo"] < 0 < res["hi"]


def test_few_runs_are_never_called_a_difference() -> None:
    assert stats.compare([80.0], [95.0])["verdict"] == "not enough runs"
    assert stats.compare([80.0, 81.0], [95.0])["verdict"] == "not enough runs"


def test_identical_runs_have_a_zero_width_interval() -> None:
    res = stats.compare([80.0, 80.0, 80.0], [90.0, 90.0, 90.0])
    assert res["verdict"] == "better" and res["diff"] == 10.0
    same = stats.compare([80.0, 80.0, 80.0], [80.0, 80.0, 80.0])
    assert same["verdict"] == "within noise"


def test_bootstrap_is_used_with_many_runs_and_ignores_an_outlier() -> None:
    a = noisy(80, 1.5, 12, 8)
    b = noisy(80, 1.5, 12, 9)
    b[3] = 20.0  # one run where the engine stalled
    res = stats.compare(a, b)
    assert res["method"] == "bootstrap"
    assert res["verdict"] == "within noise"


def test_minimum_effect_turns_a_tiny_certain_difference_into_noise() -> None:
    a = [80.0 + 0.01 * i for i in range(10)]
    b = [80.8 + 0.01 * i for i in range(10)]
    assert stats.compare(a, b)["verdict"] == "better"
    assert stats.compare(a, b, min_effect_pct=3)["verdict"] == "within noise"


def test_results_do_not_change_between_calls() -> None:
    a, b = noisy(80, 3, 10, 10), noisy(83, 3, 10, 11)
    assert stats.compare(a, b) == stats.compare(a, b)


def test_aba_finds_a_real_gain() -> None:
    res = stats.aba(noisy(80, 1.5, 4, 12), noisy(92, 1.5, 4, 13), noisy(80, 1.5, 4, 14))
    assert res["verdict"] == "better" and res["drift"] is False


def test_aba_reports_no_gain_as_within_noise() -> None:
    res = stats.aba(noisy(80, 4, 4, 15), noisy(80, 4, 4, 16), noisy(80, 4, 4, 17))
    assert res["verdict"] == "within noise" and res["within_noise"] is True


def test_aba_does_not_trust_a_gain_that_is_only_drift() -> None:
    # The machine slows down during the test: A1 is fast (100), A2 slow (90), B (101) is as fast as the fast end.
    # B beats the pooled A, but it is not better than A1: the "gain" is the drift between the A blocks.
    a1 = noisy(100, 1, 4, 3)
    a2 = noisy(90, 1, 4, 103)
    b = noisy(101, 3, 4, 203)
    res = stats.aba(a1, b, a2)
    assert res["drift"] is True
    assert res["vs_a1"]["verdict"] == "within noise" and res["vs_a2"]["verdict"] == "better"
    assert res["verdict"] == "within noise"
    assert "drift" in res["reason"]


def test_aba_with_one_run_each_says_so() -> None:
    assert stats.aba([80.0], [90.0], [81.0])["verdict"] == "not enough runs"


def test_t_interval_matches_a_hand_calculation() -> None:
    lo, hi = stats.t_interval([10.0, 12.0, 14.0]) or (0, 0)
    half = stats.t_quantile(0.975, 2) * 2.0 / math.sqrt(3)
    assert (lo, hi) == pytest.approx((12 - half, 12 + half))
