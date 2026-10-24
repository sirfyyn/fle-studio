"""Small statistics for comparing speed measurements, standard library only.

Why this exists: a decode speed measured once is a sample, not a fact. Sampling, background load and the engine's own
threads move it from run to run, so "B is 3 % faster than A" means nothing unless the runs agree. This module gives
the two things FLE Studio needs:

* ``summarize(values)``: the median of repeated runs with a 95 % interval;
* ``compare(a, b)`` and ``aba(a1, b, a2)``: the difference between two settings with a 95 % interval, and a verdict
  that says "faster" or "slower" only when that interval excludes 0, otherwise "within noise".

Two interval methods, picked by the number of runs (``method`` in every result):

* ``welch``: Welch's t interval of the difference of the means (the one to use with few runs; a bootstrap of 3 values
  can only return values between the smallest and the largest, which would claim far more certainty than there is);
* ``bootstrap``: percentile bootstrap of the difference of the medians, from 8 runs per group on. Medians shrug off the
  odd outlier run that a mean would follow.

All random draws use a fixed seed, so the same numbers always give the same verdict.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import Any

BOOTSTRAP_MIN = 8  # runs per group from which the bootstrap is used
BOOTSTRAP_DRAWS = 2000
SEED = 20261011

# ------------------------------------------------------------------------------------------------ t distribution


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Cumulative distribution function of Student's t with ``df`` degrees of freedom (fractions allowed)."""
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_quantile(p: float, df: float) -> float:
    """The t value with ``t_cdf(t, df) == p`` (bisection; exact to about 1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be between 0 and 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e9:
            break
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ------------------------------------------------------------------------------------------------ one sample


def _clean(values: Sequence[float | None]) -> list[float]:
    return [float(v) for v in values if v is not None and not math.isnan(float(v))]


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def t_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float] | None:
    """Student's t interval of the mean; None for fewer than 2 values."""
    xs = _clean(values)
    if len(xs) < 2:
        return None
    mean = statistics.fmean(xs)
    sem = statistics.stdev(xs) / math.sqrt(len(xs))
    q = t_quantile(0.5 + level / 2.0, len(xs) - 1)
    return mean - q * sem, mean + q * sem


def bootstrap_interval(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = median,
    level: float = 0.95,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval of ``stat`` (the median by default); None for fewer than 2 values."""
    xs = _clean(values)
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    n = len(xs)
    stats = sorted(stat([xs[rng.randrange(n)] for _ in range(n)]) for _ in range(draws))
    alpha = (1.0 - level) / 2.0
    return stats[int(alpha * draws)], stats[min(draws - 1, int((1.0 - alpha) * draws))]


def summarize(values: Sequence[float | None], level: float = 0.95) -> dict[str, Any]:
    """Median and 95 % interval of repeated runs.

    ``lo``/``hi`` are None when there are fewer than 3 runs: two runs say nothing about the spread.
    """
    xs = _clean(values)
    out: dict[str, Any] = {"n": len(xs), "median": None, "mean": None, "lo": None, "hi": None, "method": None}
    if not xs:
        return out
    out["median"] = round(median(xs), 4)
    out["mean"] = round(statistics.fmean(xs), 4)
    if len(xs) >= 3:
        if len(xs) >= BOOTSTRAP_MIN:
            ci, out["method"] = bootstrap_interval(xs, level=level), "bootstrap"
        else:
            ci, out["method"] = t_interval(xs, level), "t"
        if ci:
            out["lo"], out["hi"] = round(ci[0], 4), round(ci[1], 4)
    return out


# ------------------------------------------------------------------------------------------------ two samples


def difference_interval(
    a: Sequence[float], b: Sequence[float], level: float = 0.95, seed: int = SEED
) -> tuple[float, float, str] | None:
    """Interval of ``b - a`` and the method used; None when a group has fewer than 2 runs."""
    xa, xb = _clean(a), _clean(b)
    if len(xa) < 2 or len(xb) < 2:
        return None
    if min(len(xa), len(xb)) >= BOOTSTRAP_MIN:
        rng = random.Random(seed)
        na, nb = len(xa), len(xb)
        diffs = sorted(
            median([xb[rng.randrange(nb)] for _ in range(nb)]) - median([xa[rng.randrange(na)] for _ in range(na)])
            for _ in range(BOOTSTRAP_DRAWS)
        )
        alpha = (1.0 - level) / 2.0
        return (
            diffs[int(alpha * BOOTSTRAP_DRAWS)],
            diffs[min(BOOTSTRAP_DRAWS - 1, int((1 - alpha) * BOOTSTRAP_DRAWS))],
            "bootstrap",
        )
    va, vb = statistics.variance(xa) / len(xa), statistics.variance(xb) / len(xb)
    diff = statistics.fmean(xb) - statistics.fmean(xa)
    se = math.sqrt(va + vb)
    if se == 0.0:
        return diff, diff, "welch"
    df = (va + vb) ** 2 / ((va**2) / (len(xa) - 1) + (vb**2) / (len(xb) - 1))
    q = t_quantile(0.5 + level / 2.0, df)
    return diff - q * se, diff + q * se, "welch"


def compare(
    a: Sequence[float | None],
    b: Sequence[float | None],
    *,
    higher_is_better: bool = True,
    min_effect_pct: float = 0.0,
    level: float = 0.95,
) -> dict[str, Any]:
    """Is B different from A? ``diff`` is ``B - A`` in the unit of the values, ``pct`` relative to A.

    verdict: "better" / "worse" (in the sense of ``higher_is_better``) only when the interval excludes 0 and the
    effect is at least ``min_effect_pct`` percent; "within noise" when it does not; "not enough runs" for fewer than 2
    runs per group. ``a`` is the reference the percentages refer to.
    """
    xa, xb = _clean(a), _clean(b)
    out: dict[str, Any] = {
        "n_a": len(xa),
        "n_b": len(xb),
        "a": summarize(xa, level),
        "b": summarize(xb, level),
        "diff": None,
        "lo": None,
        "hi": None,
        "pct": None,
        "pct_lo": None,
        "pct_hi": None,
        "method": None,
        "verdict": "not enough runs",
        "within_noise": None,
    }
    ci = difference_interval(xa, xb, level)
    if ci is None:
        return out
    lo, hi, method = ci
    base = median(xa) if method == "bootstrap" else statistics.fmean(xa)
    center = (median(xb) - median(xa)) if method == "bootstrap" else (statistics.fmean(xb) - statistics.fmean(xa))
    out.update(diff=round(center, 4), lo=round(lo, 4), hi=round(hi, 4), method=method)
    if base:
        out.update(
            pct=round(100 * center / base, 2), pct_lo=round(100 * lo / base, 2), pct_hi=round(100 * hi / base, 2)
        )
    excludes_zero = lo > 0 or hi < 0
    big_enough = base == 0 or min(abs(100 * lo / base), abs(100 * hi / base)) >= min_effect_pct
    if excludes_zero and big_enough:
        up = lo > 0
        out["verdict"] = "better" if up == higher_is_better else "worse"
        out["within_noise"] = False
    else:
        out["verdict"] = "within noise"
        out["within_noise"] = True
    return out


def aba(
    a1: Sequence[float | None],
    b: Sequence[float | None],
    a2: Sequence[float | None],
    *,
    higher_is_better: bool = True,
    min_effect_pct: float = 0.0,
) -> dict[str, Any]:
    """The A/B/A design: A, then B, then A again, with the same number of runs each.

    Measuring A before and after B exposes drift (thermal throttling, background load, a cache that warms up): if the
    two A blocks differ from each other, a difference between A and B may just be that drift. Rules:

    * B is compared with both A blocks pooled. The speed difference counts only if its interval excludes 0
      ("faster" / "slower"); otherwise the verdict is "within noise".
    * B must also lie on the same side of A1 and of A2. If it does not, the verdict is "within noise" even when the
      pooled interval excludes 0.
    * ``drift`` is True when the two A blocks differ significantly. Then both single comparisons must be significant,
      or the verdict falls back to "within noise".
    """
    pooled = _clean(a1) + _clean(a2)
    main = compare(pooled, b, higher_is_better=higher_is_better, min_effect_pct=min_effect_pct)
    vs1 = compare(a1, b, higher_is_better=higher_is_better, min_effect_pct=min_effect_pct)
    vs2 = compare(a2, b, higher_is_better=higher_is_better, min_effect_pct=min_effect_pct)
    drift_cmp = compare(a1, a2, higher_is_better=higher_is_better)
    drift = drift_cmp["within_noise"] is False
    verdict = main["verdict"]
    reason = ""
    if verdict in ("better", "worse"):
        if (vs1["diff"] or 0) * (vs2["diff"] or 0) <= 0:
            verdict, reason = "within noise", "B does not lie on the same side of both A blocks"
        elif drift and not (vs1["verdict"] == verdict and vs2["verdict"] == verdict):
            verdict, reason = "within noise", "the two A blocks differ (drift), so the pooled difference is not trusted"
    if verdict == "within noise" and main["verdict"] == "not enough runs":
        verdict = "not enough runs"
    main.update(
        verdict=verdict,
        within_noise=verdict == "within noise",
        drift=drift,
        drift_pct=drift_cmp["pct"],
        vs_a1=vs1,
        vs_a2=vs2,
        reason=reason,
    )
    return main
