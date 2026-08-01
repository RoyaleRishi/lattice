"""Confidence intervals over resampling draws. Percentile and BCa
(bias-corrected and accelerated) for n-out-of-n bootstrap draws; a rescaled
construction for m-out-of-n subsampling draws; paired delta for comparative
claims. BCa is only valid for the "resample" scheme — it assumes the draws are
a bootstrap distribution at the full sample size. Stdlib only —
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


def subsample_interval(
    estimate: float,
    subsamples: list[float],
    *,
    m: int,
    n: int,
    level: float = 0.95,
) -> Interval:
    """Percentile-form m-out-of-n subsampling interval. The draws come from size-m
    replicates, which are more variable than the full size-n sample by the
    convergence rate sqrt(n / m); the correction is to lay the draw quantiles
    back around the point estimate shrunk by tau = sqrt(m / n):

        lo = estimate + tau * (q(alpha/2)     - estimate)
        hi = estimate + tau * (q(1 - alpha/2) - estimate)

    ATTRIBUTION — this is NOT the interval in Politis, Romano & Wolf,
    *Subsampling* (1999), and should not be checked against it as if it were.
    PRW's construction is root-based: it estimates the sampling distribution of
    sqrt(m) * (theta_hat_m - theta_hat_n) and inverts it, which *reflects* the
    quantiles about the estimate —

        lo = estimate - tau * (q(1 - alpha/2) - estimate)
        hi = estimate - tau * (q(alpha/2)     - estimate)

    — where the form above re-centres them without reflecting. The two coincide
    only when the draw distribution is symmetric about the estimate; on
    one-sided draws they move in opposite directions. What this function borrows
    from PRW is the sqrt(m / n) rate, not the interval.

    Why the percentile form was chosen here: on the size-dependent pooled
    metrics this codebase actually resamples (see below), the draws are
    systematically to one side of the estimate. The reflected form then throws
    the band across to the *opposite* side — on M4 food, where every draw sits
    below the estimate, PRW's interval would sit entirely above it — which is
    strictly less informative than a band that shrinks toward the estimate from
    the side the draws are on. This is a deliberate choice for one-sided
    behaviour, not a transcription of the reference.

    Unlike a raw percentile interval over subsample draws this is anchored on
    the estimate, so it brackets it whenever the draws straddle it, and when
    they all sit to one side it shrinks toward the estimate instead of
    reporting a band that excludes it. tau == 1 (m == n) reduces exactly to the
    percentile interval.

    IMPORTANT — when the result is not a confidence interval. Subsampling
    assumes theta_hat_m and theta_hat_n are centred on the same theta, so that
    the only difference between a size-m replicate and the full sample is a
    known rate. That assumption fails for a *size-dependent* functional: one
    whose value depends on how many distinct documents are in the sample at
    all, either because its prediction side is a set union over documents (so
    it grows sublinearly and never reaches the full-corpus value) or because
    its denominator is a fixed corpus-level quantity that does not shrink with
    the sample. For such a metric the draws sit systematically to one side of
    the estimate and never straddle it, at any m, under this scheme *and*
    under the with-replacement bootstrap alike — it is a bias in the
    functional, not variance that a rate correction can rescale away. The
    output is then a corpus-size sensitivity range, not a confidence interval,
    and must not be quoted as one.

    Size dependence is a spectrum, not a binary. `edge-f1` is the severe
    shipped case — its bands never bracket. `clustering`'s B³ keys are a
    milder, opposite-signed case that brackets only because tau-shrinkage pulls
    the one-sided draws back far enough. Both metrics document their own
    behaviour; read theirs before quoting a band. report.py emits
    `brackets_estimate`, which is decisive when False but is necessary rather
    than sufficient when True.

    n == 0 means every document was held fixed: there is nothing to resample,
    so the honest answer is a zero-width interval at the estimate.
    """
    if not subsamples:
        raise ValueError("subsample_interval requires at least one draw")
    if n == 0:
        return Interval(estimate, estimate, "degenerate")
    if not 0 < m <= n:
        raise ValueError(f"subsample_interval requires 0 < m <= n (got m={m}, n={n})")
    s = sorted(subsamples)
    a = (1 - level) / 2
    tau = math.sqrt(m / n)
    lo = estimate + tau * (_percentile(s, a) - estimate)
    hi = estimate + tau * (_percentile(s, 1 - a) - estimate)
    return Interval(lo, hi, "subsample")


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
    if len(resamples_a) != len(resamples_b):
        raise ValueError(
            "paired_delta requires equal-length resample lists "
            f"(got {len(resamples_a)} and {len(resamples_b)})"
        )
    deltas = [x - y for x, y in zip(resamples_a, resamples_b)]
    iv = percentile_interval(estimate_a - estimate_b, deltas, level)
    prob = sum(1 for d in deltas if d > 0) / len(deltas)
    return DeltaResult(estimate_a - estimate_b, iv.lo, iv.hi, prob)
