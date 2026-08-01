"""Does the m-out-of-n subsampling interval actually cover at its nominal
level? Nothing else in the suite asks. Every other subsample test pins the
*arithmetic* — hand-computed quantiles, guard behaviour, bit-identity under a
seed — and all of them passed for the whole life of the sqrt(m / n) factor,
which produced nominal-95% bands that covered about 83%. Arithmetic tests
cannot catch a wrong formula, only a wrongly-executed one. This file is the
one that can.

The fixture is the mean of iid normals, chosen because it removes everything
except the factor: it is exactly root-n consistent, symmetric, and has no size
dependence whatsoever, so theta(m) and theta(n) are centred on the same theta
and any coverage miss is the rescaling and nothing else. (The pooled metrics
this library ships are *not* like that — see subsample_interval's docstring on
size-dependent functionals — which is precisely why coverage has to be
measured on a clean case rather than on them.)

GC3: fully seeded. random.Random(seed) with a fixed trial/draw count makes
every number below deterministic across runs and machines, so the assertions
are on a fixed measurement, not on a random one that happens to pass.
"""

import math
import random

import pytest

from lattice.harness.stats.intervals import _subsample_tau, subsample_interval

TRUE_MEAN = 0.0
SIGMA = 1.0
LEVEL = 0.95

# Sized for a decision, not for a publishable coverage estimate. At 400 trials
# the standard error on a coverage near 0.95 is sqrt(.95*.05/400) = 0.011, so
# the 0.92-0.97 band below is +-2.3 se — wide enough not to be flaky in
# principle, and moot in practice because the seed pins the outcome. The whole
# module runs in a couple of seconds.
TRIALS = 400
DRAWS = 200


def _coverage(n: int, m: int, seed: int, *, report_n: int | None = None) -> float:
    """Fraction of TRIALS whose emitted interval contains the true mean.

    `report_n` exists to measure a *different* tau through the real
    subsample_interval instead of reimplementing its arithmetic here: the draws
    are always m-out-of-n from the true pool, but the interval is told the pool
    is `report_n`, and tau is sqrt(m / (report_n - m)). So report_n = n + m
    yields sqrt(m / n) — exactly the pre-2026-08-01 factor — and report_n = 2m
    yields a constant 1, both over the identical draws.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(TRIALS):
        sample = [rng.gauss(TRUE_MEAN, SIGMA) for _ in range(n)]
        estimate = sum(sample) / n
        draws = [sum(rng.sample(sample, m)) / m for _ in range(DRAWS)]
        iv = subsample_interval(estimate, draws, m=m, n=report_n or n, level=LEVEL)
        hits += iv.lo <= TRUE_MEAN <= iv.hi
    return hits / TRIALS


def test_the_substitute_factors_are_reachable_through_the_public_function():
    # Guards the trick _coverage uses for its controls. tau is
    # sqrt(m / (report_n - m)), so report_n = n + m yields the pre-10c
    # sqrt(m / n) and report_n = 2m yields a constant 1. If either identity
    # ever stops holding, the control tests below would silently be measuring
    # the corrected factor again and would pass vacuously.
    for n, m in ((60, 30), (200, 100), (200, 50)):
        assert _subsample_tau(m, n) == pytest.approx(math.sqrt(m / (n - m)))
        assert _subsample_tau(m, n + m) == pytest.approx(math.sqrt(m / n))
        assert _subsample_tau(m, 2 * m) == 1.0


@pytest.mark.parametrize(
    ("n", "m", "seed"),
    [
        (60, 30, 20260801),    # the harness's own draw size; tau == 1
        (200, 100, 20260802),  # ditto, larger pool
        (200, 50, 20260805),   # m = n / 4, where tau = sqrt(1/3) = 0.577 != 1
    ],
)
def test_subsample_interval_covers_at_its_nominal_level(n, m, seed):
    # The nominal level is 0.95; assert a band rather than a point because
    # Monte Carlo coverage at 400 trials is an estimate and the interval is
    # asymptotic.
    #
    # The third case is load-bearing and not redundant. At m = n / 2 the
    # correct tau is exactly 1, so coverage there cannot distinguish
    # sqrt(m / (n - m)) from ANY expression that happens to equal 1 at the
    # half point — a hardcoded 1.0, m / (n - m) without the square root, even
    # the inverted sqrt((n - m) / m). Only a draw size away from n / 2
    # constrains the shape of the factor rather than one of its values.
    coverage = _coverage(n, m, seed)
    assert 0.92 <= coverage <= 0.97, (
        f"n={n}, m={m}: coverage {coverage:.3f} outside [0.92, 0.97]"
    )


@pytest.mark.parametrize(("n", "seed"), [(60, 20260801), (200, 20260802)])
def test_the_pre_10c_factor_undercovers_on_the_same_draws(n, seed):
    # The defect, pinned. Same seed and therefore the identical draws as the
    # test above — the only thing that changes is tau, from sqrt(m / (n - m))
    # = 1 to sqrt(m / n) = 0.707, a band 29% too narrow. Nominal 95% lands
    # near 83%. Without this assertion the fix has no evidence attached to it;
    # with it, reverting the factor fails loudly here rather than quietly
    # publishing an over-confident CI.
    #
    # Deliberately only at m = n / 2, which is where the harness draws and
    # where the two factors are furthest apart. At m = n / 4 the old factor is
    # 0.500 against a correct 0.577 — only 13% narrow — and it lands around
    # 0.90-0.93, i.e. sometimes inside the band above. The old factor is not
    # uniformly catastrophic; it is catastrophic at the draw size this project
    # actually uses, and that is the claim worth pinning.
    m = round(n / 2)
    old = _coverage(n, m, seed, report_n=n + m)
    assert old < 0.90, f"n={n}: sqrt(m/n) coverage {old:.3f} was expected to undercover"
    assert old < _coverage(n, m, seed)


def test_a_constant_tau_overcovers_away_from_the_half_point():
    # The other side of the shape constraint. tau == 1 is right at m = n / 2
    # and wrong everywhere else; at m = n / 4 the draws are genuinely more
    # dispersed than the estimate and leaving them unscaled inflates the band.
    # Nominal 95% becomes ~0.998 — an interval that is not wrong in the
    # dangerous direction, but is not a 95% interval either.
    n, m, seed = 200, 50, 20260805
    flat = _coverage(n, m, seed, report_n=2 * m)
    assert flat > 0.99, f"constant tau=1 coverage {flat:.3f} was expected to over-cover"
    assert flat > _coverage(n, m, seed)
