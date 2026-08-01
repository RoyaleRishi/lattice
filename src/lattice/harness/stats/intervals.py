"""Confidence intervals over resampling draws. Percentile and BCa
(bias-corrected and accelerated) for n-out-of-n bootstrap draws; a rescaled
construction for m-out-of-n subsampling draws; paired delta for comparative
claims, which takes either scheme. BCa is only valid for the "resample" scheme
— it assumes the draws are a bootstrap distribution at the full sample size,
so it is offered neither by subsample_interval nor by paired_delta. Stdlib
only — statistics.NormalDist supplies the normal CDF and its inverse."""

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


def _subsample_tau(m: int, n: int) -> float | None:
    """The rescaling factor for m-out-of-n subsampling *without replacement*.
    Single source of truth: both subsample_interval and paired_delta read it
    from here so a band and its prob_positive can never be scaled differently.

    A size-m replicate shares m of the full sample's n documents, so the spread
    of (theta_m - theta_n) is not the spread of theta_m — the shared documents
    cancel. For a root-n-consistent functional with per-document variance
    sigma^2:

        Var(theta_m - theta_n) = sigma^2/m - sigma^2/n = (sigma^2/n)(n - m)/m
        Var(theta_n - theta)   = sigma^2/n
        ratio                  = m / (n - m)

    so the draw quantiles carry sqrt((n - m)/m) times the spread the estimate
    itself has, and the correction is tau = sqrt(m / (n - m)).

    NOT sqrt(m / n), which this function used before 2026-08-01. That is the
    same factor with the finite-population term (n - m)/n dropped, legitimate
    only in the b/n -> 0 asymptotic regime — which is not the regime here:
    resample.subsample_size() draws m = round(n / 2), where the overlap with
    the full sample is half the corpus and the dropped term is the dominant
    one. At exactly m = n / 2 the correct tau is 1.0 while sqrt(m / n) is
    0.707, so the old bands were 29% too narrow. Measured on the mean of iid
    normals, nominal 95%: coverage ~0.83 under sqrt(m / n), ~0.95 under
    sqrt(m / (n - m)). test_subsample_coverage.py measures both and is the
    regression pin.

    Returns None when there is nothing to rescale, which callers must render as
    a degenerate zero-width band rather than dividing by zero:

    - n == 0: every document was held fixed, so nothing was resampled.
    - m >= n: the "subsample" is the whole pool, so every replicate is the same
      document set. Reachable, not hypothetical — subsample_size() floors m at
      2, so n == 2 yields m == n.
    """
    if n <= 0 or m >= n:
        return None
    return math.sqrt(m / (n - m))


def subsample_interval(
    estimate: float,
    subsamples: list[float],
    *,
    m: int,
    n: int,
    level: float = 0.95,
) -> Interval:
    """Percentile-form m-out-of-n subsampling interval. The draws come from size-m
    replicates drawn without replacement, which vary about the full-sample
    estimate by a known factor; the correction is to lay the draw quantiles back
    around the point estimate rescaled by tau = sqrt(m / (n - m)):

        lo = estimate + tau * (q(alpha/2)     - estimate)
        hi = estimate + tau * (q(1 - alpha/2) - estimate)

    See _subsample_tau for the derivation, and for why the sqrt(m / n) this
    function used before 2026-08-01 is the wrong factor at the m = round(n / 2)
    the harness actually draws at.

    WHAT TAU DOES AT THE HARNESS'S DRAW SIZE — read this before relying on the
    anchoring behaviour described below. tau is a shrink factor only for
    m < n / 2. At m = n / 2 it is exactly 1, and this whole construction is
    then the *identity*: lo and hi are the raw draw quantiles, and the
    anchoring on the estimate does nothing at all. For m > n / 2, tau exceeds
    1 and the band is wider than the draws. resample.subsample_size() takes
    m = round(n / 2), so on even n the emitted band is exactly the percentile
    band over the size-m draws, and on odd n it is within about 1/n of it.
    Under the old sqrt(m / n) the anchoring was always doing visible work at
    that draw size; it no longer is, which changes how the one-sided pooled
    metrics below behave and is why their measured figures need regenerating.

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
    from PRW is the idea of a rate correction, not the interval.

    The rate is no longer a departure from PRW, though it was. PRW state their
    factor for b/n -> 0, where the finite-population term (n - b)/n vanishes
    and the factor collapses to sqrt(b / n); at b = n / 2 that term is the
    dominant one and cannot be dropped. tau = sqrt(m / (n - m)) is PRW's factor
    with the term kept, and reduces to their sqrt(m / n) as m / n -> 0. So the
    remaining departure from the reference is in the interval *form* only.

    Why the percentile form was chosen here: on the size-dependent pooled
    metrics this codebase actually resamples (see below), the draws are
    systematically to one side of the estimate. The reflected form then throws
    the band across to the *opposite* side — on M4 food, where every draw sits
    below the estimate, PRW's interval would sit entirely above it — which is
    strictly less informative than a band that stays on the side the draws are
    actually on. That argument is about the reflection, not the magnitude of
    tau, so it survives tau == 1 unchanged. This is a deliberate choice for
    one-sided behaviour, not a transcription of the reference.

    Unlike a raw percentile interval over subsample draws this is anchored on
    the estimate, so at m < n / 2 it brackets whenever the draws straddle, and
    when they all sit to one side it shrinks toward the estimate instead of
    reporting a band that excludes it. At m = n / 2 that anchoring is inert
    (tau == 1, see above): a one-sided draw distribution then yields a band
    that simply excludes the estimate, and `brackets_estimate` says so.

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
    milder, opposite-signed case whose draws sit above the estimate but still
    overlap it. Under the pre-2026-08-01 sqrt(m / n) that overlap was partly
    manufactured by shrinkage; at the corrected tau = 1 whether a B³ band
    brackets is a property of the draws alone. Both metrics document their own
    behaviour; read theirs before quoting a band. report.py emits
    `brackets_estimate`, which is decisive when False but is necessary rather
    than sufficient when True.

    Degenerate cases (see _subsample_tau): n == 0 means every document was held
    fixed, and m >= n means each replicate is the whole pool. Either way there
    is no resampling variability, so the honest answer is a zero-width interval
    at the estimate — not a division by zero.
    """
    if not subsamples:
        raise ValueError("subsample_interval requires at least one draw")
    if n == 0:
        return Interval(estimate, estimate, "degenerate")
    if not 0 < m <= n:
        raise ValueError(f"subsample_interval requires 0 < m <= n (got m={m}, n={n})")
    tau = _subsample_tau(m, n)
    if tau is None:
        return Interval(estimate, estimate, "degenerate")
    s = sorted(subsamples)
    a = (1 - level) / 2
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
    *, m: int | None = None, n: int | None = None,
) -> DeltaResult:
    """Interval on the difference of two statistics, paired iteration by
    iteration. Returns the observed delta, its interval, and the fraction of
    replicate deltas above zero.

    PAIRING INVARIANT — load-bearing, and not checkable from in here. Element i
    of `resamples_a` and element i of `resamples_b` must come from the *same*
    replicate document set; that is the whole reason a paired delta is tighter
    than the difference of two marginal intervals. The library gets it by
    construction rather than by assertion: resample.bootstrap() builds a fresh
    random.Random(seed) per call and walks an insertion-ordered pool, so two
    calls at the same seed, over two bundles whose `per_document` holds the
    same document ids in the same order, consume the identical RNG sequence and
    therefore draw the identical indices at every iteration — under "resample"
    and "subsample" alike. Callers must pass both bundles the same seed and
    must not reorder either bundle's `per_document` between the two calls. The
    only thing this function can check is that the two lists are the same
    length; equal length with misaligned draws silently yields a wrong
    (typically too wide) interval, so callers over two separately-built bundles
    should compare the two document-id lists themselves.

    SCHEME. With `m` and `n` omitted the draws are read as an n-out-of-n
    bootstrap and the interval is the plain percentile interval over the paired
    differences — the classical path, unchanged. Pass `m` and `n` (straight off
    the BootstrapDraws) when the draws came from m-out-of-n subsampling, which
    is mandatory for pooled metrics (see resample.py): the delta distribution
    is then rescaled by tau = sqrt(m / (n - m)) about the *observed* delta,
    exactly as subsample_interval() rescales a single statistic about its own
    estimate. Reading size-m deltas as if they were full-size bootstrap deltas
    misstates the spread by sqrt((n - m) / m). Passing one of `m`/`n` without
    the other is an error rather than a silent fallback to the wrong scheme.

    `prob_positive` is computed from the same distribution the interval is: the
    raw paired differences on the classical path, the tau-rescaled ones under
    subsampling. Otherwise a subsample interval could exclude zero while
    `prob_positive` reported appreciable mass on the other side of it. Both the
    interval and that rescaling take tau from the same _subsample_tau(), so
    they cannot drift apart: a duplicated `math.sqrt(m / n)` on this line is
    precisely how the pre-2026-08-01 factor would survive here after being
    corrected in subsample_interval, and must not be reintroduced.

    BCa is deliberately not offered here under either scheme.

    A paired delta of two *size-dependent* functionals can be better behaved
    than either marginal: both arms are scored on the same m documents, so a
    size effect common to the two arms cancels in the difference (B3 on M3 is
    the live example — theta(29) > theta(58) for both resolvers). That is a
    property of the particular pair, not a licence: it holds only to the extent
    the two arms share the size effect, which is an empirical question per
    comparison and not something this function can guarantee.
    """
    if len(resamples_a) != len(resamples_b):
        raise ValueError(
            "paired_delta requires equal-length resample lists "
            f"(got {len(resamples_a)} and {len(resamples_b)})"
        )
    if (m is None) != (n is None):
        raise ValueError(
            "paired_delta requires both m and n for the subsample scheme, or neither "
            f"for the n-out-of-n bootstrap (got m={m}, n={n})"
        )
    observed = estimate_a - estimate_b
    deltas = [x - y for x, y in zip(resamples_a, resamples_b)]
    if m is None or n is None:
        iv = percentile_interval(observed, deltas, level)
        scaled = deltas
    else:
        iv = subsample_interval(observed, deltas, m=m, n=n, level=level)
        # Same helper the interval used, so prob_positive is read off the same
        # distribution the band was built from. None is the degenerate case
        # (n == 0, everything held fixed; or m >= n, every replicate the whole
        # pool): subsample_interval collapses the band onto the observed delta,
        # so the distribution must collapse with it or the two would contradict
        # each other.
        tau = _subsample_tau(m, n)
        scaled = (
            [observed] * len(deltas)
            if tau is None
            else [observed + tau * (d - observed) for d in deltas]
        )
    prob = sum(1 for d in scaled if d > 0) / len(scaled)
    return DeltaResult(observed, iv.lo, iv.hi, prob)
