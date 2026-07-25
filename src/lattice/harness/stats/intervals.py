"""Bootstrap confidence intervals. Percentile and BCa (bias-corrected and
accelerated); paired delta for comparative claims. Stdlib only —
statistics.NormalDist supplies the normal CDF and its inverse."""

import math
from dataclasses import dataclass
from statistics import NormalDist

_N = NormalDist()


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    method: str


@dataclass(frozen=True)
class DeltaResult:
    estimate: float
    lo: float
    hi: float
    prob_positive: float


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated quantile (numpy 'linear' method): position q*(n-1)."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[int(pos)]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def percentile_interval(estimate: float, resamples: list[float], level: float = 0.95) -> Interval:
    s = sorted(resamples)
    a = (1 - level) / 2
    return Interval(_percentile(s, a), _percentile(s, 1 - a), "percentile")


def bca_interval(
    estimate: float, resamples: list[float], jackknife: list[float], level: float = 0.95
) -> Interval:
    s = sorted(resamples)
    a = (1 - level) / 2
    if s[0] == s[-1]:
        return Interval(estimate, estimate, "degenerate")
    prop = sum(1 for r in resamples if r < estimate) / len(resamples)
    if prop <= 0.0 or prop >= 1.0:
        return Interval(_percentile(s, a), _percentile(s, 1 - a), "percentile-fallback")
    z0 = _N.inv_cdf(prop)
    jbar = sum(jackknife) / len(jackknife)
    num = sum((jbar - j) ** 3 for j in jackknife)
    den = 6.0 * (sum((jbar - j) ** 2 for j in jackknife)) ** 1.5
    acc = num / den if den != 0 else 0.0
    bounds = []
    for z_a in (_N.inv_cdf(a), _N.inv_cdf(1 - a)):
        adj = z0 + (z0 + z_a) / (1 - acc * (z0 + z_a))
        bounds.append(_percentile(s, _N.cdf(adj)))
    lo, hi = bounds
    if lo > estimate or hi < estimate:
        # Extreme bias/acceleration on a heavily skewed resample distribution can
        # collapse the BCa interval to a range that does not bracket the point
        # estimate — a degenerate, misleading "CI". Fall back to the plain
        # percentile interval, which honestly reflects where the resamples lie.
        return Interval(_percentile(s, a), _percentile(s, 1 - a), "percentile-fallback")
    return Interval(lo, hi, "bca")


def paired_delta(
    resamples_a: list[float], resamples_b: list[float],
    estimate_a: float, estimate_b: float, level: float = 0.95,
) -> DeltaResult:
    deltas = [x - y for x, y in zip(resamples_a, resamples_b)]
    iv = percentile_interval(estimate_a - estimate_b, deltas, level)
    prob = sum(1 for d in deltas if d > 0) / len(deltas)
    return DeltaResult(estimate_a - estimate_b, iv.lo, iv.hi, prob)
