import pytest

from lattice.harness.stats.intervals import (
    bca_interval,
    paired_delta,
    percentile_interval,
    subsample_interval,
)


def test_percentile_linear_interpolation():
    # 0..99: numpy linear 2.5th/97.5th percentiles are 2.475 and 96.525
    iv = percentile_interval(0.0, [float(i) for i in range(100)])
    assert iv.method == "percentile"
    assert abs(iv.lo - 2.475) < 1e-9
    assert abs(iv.hi - 96.525) < 1e-9


def test_bca_reduces_to_percentile_when_symmetric():
    # estimate 0, exactly half strictly below -> z0=0; symmetric jackknife -> acc=0
    resamples = list(range(-50, 0)) + list(range(1, 51))
    resamples = [float(x) for x in resamples]
    iv = bca_interval(0.0, resamples, [1.0, -1.0, 2.0, -2.0, 0.0])
    pv = percentile_interval(0.0, resamples)
    assert iv.method == "bca"
    assert abs(iv.lo - pv.lo) < 1e-9 and abs(iv.hi - pv.hi) < 1e-9


def test_bca_degenerate_zero_width():
    iv = bca_interval(5.0, [5.0] * 20, [5.0] * 4)
    assert (iv.lo, iv.hi, iv.method) == (5.0, 5.0, "degenerate")


def test_bca_falls_back_when_estimate_outside_resamples():
    iv = bca_interval(0.0, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert iv.method == "percentile-fallback"


def test_bca_falls_back_when_interval_would_not_bracket_estimate():
    # Heavily right-skewed resamples with the estimate near the 1st percentile drive
    # BCa's bias/acceleration adjustment to collapse the interval below the estimate
    # (raw BCa here is ~[0.0, 3.37], which does not bracket 5.0). The guard must fall
    # back to the percentile interval rather than report a non-bracketing "CI".
    resamples = [0.0] + [10.0] * 99
    iv = bca_interval(5.0, resamples, [1.0, 1.0, 1.0, 2.0])
    assert iv.method == "percentile-fallback"
    assert (iv.lo, iv.hi) == (10.0, 10.0)


def test_subsample_interval_brackets_estimate_when_draws_straddle_it():
    # Hand-computed. draws = [0, 1, 2, 3, 4], estimate = 2.0, m = 2, n = 8,
    # level = 0.5 so alpha/2 = 0.25.
    #   q(0.25): pos = 0.25 * (5 - 1) = 1.0        -> draws[1] = 1.0
    #   q(0.75): pos = 0.75 * (5 - 1) = 3.0        -> draws[3] = 3.0
    #   tau     = sqrt(m / n) = sqrt(2 / 8) = 0.5
    #   lo      = 2.0 + 0.5 * (1.0 - 2.0) = 1.5
    #   hi      = 2.0 + 0.5 * (3.0 - 2.0) = 2.5
    iv = subsample_interval(2.0, [0.0, 1.0, 2.0, 3.0, 4.0], m=2, n=8, level=0.5)
    assert iv.method == "subsample"
    assert (iv.lo, iv.hi) == (1.5, 2.5)
    assert iv.lo <= 2.0 <= iv.hi


def test_subsample_interval_shrinks_toward_estimate_when_draws_do_not_straddle():
    # The concrete M4 failure mode: every draw sits above the point estimate,
    # so the raw percentile interval [3.5, 4.5] excludes the estimate entirely.
    # Rescaling by tau pulls the band toward the estimate instead of reporting
    # a "CI" centred on the draws.
    #   draws = [3, 4, 5], estimate = 1.0, m = 1, n = 4, level = 0.5
    #   q(0.25): pos = 0.5 -> 3.0 * 0.5 + 4.0 * 0.5 = 3.5
    #   q(0.75): pos = 1.5 -> 4.0 * 0.5 + 5.0 * 0.5 = 4.5
    #   tau = sqrt(1 / 4) = 0.5
    #   lo  = 1.0 + 0.5 * (3.5 - 1.0) = 2.25
    #   hi  = 1.0 + 0.5 * (4.5 - 1.0) = 2.75
    draws = [3.0, 4.0, 5.0]
    iv = subsample_interval(1.0, draws, m=1, n=4, level=0.5)
    pct = percentile_interval(1.0, draws, level=0.5)
    assert (iv.lo, iv.hi) == (2.25, 2.75)
    assert (pct.lo, pct.hi) == (3.5, 4.5)
    assert iv.lo > 1.0                       # honest: it still does not bracket
    assert iv.lo < pct.lo and iv.hi < pct.hi  # but it is strictly closer


def test_subsample_interval_reduces_to_percentile_when_m_equals_n():
    draws = [float(i) for i in range(100)]
    iv = subsample_interval(50.0, draws, m=13, n=13)
    pct = percentile_interval(50.0, draws)
    assert abs(iv.lo - pct.lo) < 1e-12 and abs(iv.hi - pct.hi) < 1e-12


def test_subsample_interval_degenerate_when_nothing_was_resampled():
    # n == 0: every document was held fixed, so there is no sampling
    # variability to report — a zero-width interval at the estimate, not a crash.
    iv = subsample_interval(0.7, [0.7] * 5, m=0, n=0)
    assert (iv.lo, iv.hi, iv.method) == (0.7, 0.7, "degenerate")


@pytest.mark.parametrize(("m", "n"), [(5, 4), (0, 4), (-1, 4)])
def test_subsample_interval_rejects_impossible_draw_sizes(m, n):
    # tau = sqrt(m / n) is only a shrink factor for 0 < m <= n; anything else
    # means the caller mis-recorded the draw and must fail loudly.
    with pytest.raises(ValueError, match="0 < m <= n"):
        subsample_interval(1.0, [1.0, 2.0], m=m, n=n)


def test_subsample_interval_rejects_empty_draws():
    with pytest.raises(ValueError, match="at least one draw"):
        subsample_interval(1.0, [], m=2, n=4)


def test_paired_delta_sign_and_probability():
    d = paired_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 2.0, 3.0)
    assert d.estimate == -1.0
    assert d.prob_positive == 0.0
    assert d.lo == -1.0 and d.hi == -1.0


def test_paired_delta_rejects_unequal_length():
    # zip() would silently truncate to the shorter list and pair the wrong
    # iterations; the guard must reject instead of computing a bogus delta.
    with pytest.raises(ValueError, match="equal-length"):
        paired_delta([1.0, 2.0, 3.0], [2.0, 3.0], 2.0, 3.0)
