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
    # Hand-computed. draws = [0, 1, 2, 3, 4], estimate = 2.0, m = 2, n = 10,
    # level = 0.5 so alpha/2 = 0.25.
    #   q(0.25): pos = 0.25 * (5 - 1) = 1.0        -> draws[1] = 1.0
    #   q(0.75): pos = 0.75 * (5 - 1) = 3.0        -> draws[3] = 3.0
    #   tau     = sqrt(m / (n - m)) = sqrt(2 / 8) = 0.5
    #   lo      = 2.0 + 0.5 * (1.0 - 2.0) = 1.5
    #   hi      = 2.0 + 0.5 * (3.0 - 2.0) = 2.5
    iv = subsample_interval(2.0, [0.0, 1.0, 2.0, 3.0, 4.0], m=2, n=10, level=0.5)
    assert iv.method == "subsample"
    assert (iv.lo, iv.hi) == (1.5, 2.5)
    assert iv.lo <= 2.0 <= iv.hi


def test_subsample_interval_shrinks_toward_estimate_when_draws_do_not_straddle():
    # The concrete M4 failure mode: every draw sits above the point estimate,
    # so the raw percentile interval [3.5, 4.5] excludes the estimate entirely.
    # Rescaling by tau pulls the band toward the estimate instead of reporting
    # a "CI" centred on the draws.
    #   draws = [3, 4, 5], estimate = 1.0, m = 1, n = 5, level = 0.5
    #   q(0.25): pos = 0.5 -> 3.0 * 0.5 + 4.0 * 0.5 = 3.5
    #   q(0.75): pos = 1.5 -> 4.0 * 0.5 + 5.0 * 0.5 = 4.5
    #   tau = sqrt(1 / (5 - 1)) = 0.5
    #   lo  = 1.0 + 0.5 * (3.5 - 1.0) = 2.25
    #   hi  = 1.0 + 0.5 * (4.5 - 1.0) = 2.75
    draws = [3.0, 4.0, 5.0]
    iv = subsample_interval(1.0, draws, m=1, n=5, level=0.5)
    pct = percentile_interval(1.0, draws, level=0.5)
    assert (iv.lo, iv.hi) == (2.25, 2.75)
    assert (pct.lo, pct.hi) == (3.5, 4.5)
    assert iv.lo > 1.0                       # honest: it still does not bracket
    assert iv.lo < pct.lo and iv.hi < pct.hi  # but it is strictly closer


def test_subsample_interval_reduces_to_percentile_at_the_harness_draw_size():
    # tau = sqrt(m / (n - m)) is exactly 1 at m = n / 2, which is what
    # resample.subsample_size() draws. The correction is then a pure
    # translation onto the estimate, not a shrink — so on draws already
    # centred on the estimate the band IS the percentile band. This is the
    # property the pre-2026-08-01 sqrt(m / n) = 0.707 violated by 29%.
    draws = [float(i) for i in range(100)]
    iv = subsample_interval(49.5, draws, m=13, n=26)
    pct = percentile_interval(49.5, draws)
    assert iv.method == "subsample"
    assert abs(iv.lo - pct.lo) < 1e-12 and abs(iv.hi - pct.hi) < 1e-12


def test_subsample_interval_is_degenerate_when_the_subsample_is_the_whole_pool():
    # m == n is reachable, not hypothetical: subsample_size() floors m at 2, so
    # n == 2 yields m == 2. tau = sqrt(m / (n - m)) would divide by zero, and
    # the honest answer is that a "subsample" equal to the pool has no
    # resampling variability at all.
    iv = subsample_interval(0.4, [0.4, 0.9, 0.1], m=2, n=2)
    assert (iv.lo, iv.hi, iv.method) == (0.4, 0.4, "degenerate")


def test_subsample_interval_degenerate_when_nothing_was_resampled():
    # n == 0: every document was held fixed, so there is no sampling
    # variability to report — a zero-width interval at the estimate, not a crash.
    iv = subsample_interval(0.7, [0.7] * 5, m=0, n=0)
    assert (iv.lo, iv.hi, iv.method) == (0.7, 0.7, "degenerate")


@pytest.mark.parametrize(("m", "n"), [(5, 4), (0, 4), (-1, 4)])
def test_subsample_interval_rejects_impossible_draw_sizes(m, n):
    # tau = sqrt(m / (n - m)) is only defined for 0 < m <= n; anything else
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


def test_paired_delta_equal_length_guard_still_holds_under_the_subsample_scheme():
    # The guard must fire before any rescaling — a length mismatch is a pairing
    # bug regardless of scheme.
    with pytest.raises(ValueError, match="equal-length"):
        paired_delta([1.0, 2.0, 3.0], [2.0, 3.0], 2.0, 3.0, m=2, n=8)


def test_paired_delta_subsample_rescaling_matches_a_hand_computed_example():
    # Hand-computed, deliberately the same arithmetic as
    # test_subsample_interval_brackets_estimate_when_draws_straddle_it so the
    # two constructions can be read side by side.
    #   deltas  = a - b = [0, 1, 2, 3, 4]
    #   observed = estimate_a - estimate_b = 2.0 - 0.0 = 2.0
    #   m = 2, n = 10, level = 0.5 so alpha/2 = 0.25
    #   q(0.25): pos = 0.25 * (5 - 1) = 1.0  -> 1.0
    #   q(0.75): pos = 0.75 * (5 - 1) = 3.0  -> 3.0
    #   tau = sqrt(2 / (10 - 2)) = 0.5
    #   lo  = 2.0 + 0.5 * (1.0 - 2.0) = 1.5
    #   hi  = 2.0 + 0.5 * (3.0 - 2.0) = 2.5
    a = [0.0, 1.0, 2.0, 3.0, 4.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    d = paired_delta(a, b, 2.0, 0.0, level=0.5, m=2, n=10)
    assert d.estimate == 2.0
    assert (d.lo, d.hi) == (1.5, 2.5)
    # identical to applying subsample_interval to the delta distribution
    # directly, anchored on the observed delta
    iv = subsample_interval(2.0, [x - y for x, y in zip(a, b)], m=2, n=10, level=0.5)
    assert (d.lo, d.hi) == (iv.lo, iv.hi)


def test_paired_delta_subsample_prob_positive_uses_the_rescaled_distribution():
    # Same fixture as above. The raw deltas are [0, 1, 2, 3, 4], of which 4/5
    # are strictly positive. The tau-rescaled deltas are
    #   2.0 + 0.5 * (d - 2.0) = [1.0, 1.5, 2.0, 2.5, 3.0]
    # — all five positive, because the size-2 replicates are over-dispersed by
    # sqrt((n - m) / m) and the rescaling removes exactly that. Reporting 0.8
    # next to an interval of [1.5, 2.5] that excludes zero would be
    # self-contradictory.
    a = [0.0, 1.0, 2.0, 3.0, 4.0]
    b = [0.0] * 5
    assert paired_delta(a, b, 2.0, 0.0, level=0.5, m=2, n=10).prob_positive == 1.0
    assert paired_delta(a, b, 2.0, 0.0, level=0.5).prob_positive == 0.8   # raw, no rescale


def test_paired_delta_without_m_and_n_is_exactly_the_percentile_path():
    # The classical with-replacement path must stay bit-identical: floating
    # point makes `est + 1.0 * (q - est)` differ from `q` in the last bits, so
    # this asserts equality, not approximate equality, against
    # percentile_interval over the same paired differences.
    a = [0.1 * i for i in range(37)]
    b = [0.03 * i for i in range(37)]
    d = paired_delta(a, b, 1.7, 0.4)
    pct = percentile_interval(1.7 - 0.4, [x - y for x, y in zip(a, b)])
    assert (d.lo, d.hi) == (pct.lo, pct.hi)
    assert d.prob_positive == sum(1 for x, y in zip(a, b) if x - y > 0) / 37


def test_paired_delta_subsample_is_narrower_than_reading_the_draws_as_a_bootstrap():
    # tau = sqrt(m / (n - m)) = sqrt(25 / 75) < 1 at m < n / 2, so the
    # subsample band is strictly inside the band you get by (wrongly) reading
    # size-m deltas as full-size bootstrap deltas.
    a = [float(i) for i in range(100)]
    b = [0.0] * 100
    sub = paired_delta(a, b, 50.0, 0.0, m=25, n=100)
    raw = paired_delta(a, b, 50.0, 0.0)
    assert raw.lo < sub.lo < sub.hi < raw.hi


def test_paired_delta_prob_positive_uses_the_same_tau_as_its_interval():
    # Regression pin for the second home of the pre-2026-08-01 defect. The
    # interval comes from subsample_interval and so inherited the corrected
    # factor for free, but prob_positive was rescaled by paired_delta's own
    # inlined `math.sqrt(m / n)` and would NOT have.
    #
    # observed = 1.0, m = 8, n = 16 (the harness's m = n / 2), deltas below.
    # Correct tau = sqrt(8 / 8) = 1, so the rescaled deltas are the deltas and
    # prob_positive is just their sign: 3 of 5 above zero.
    # The old tau = sqrt(8 / 16) = 0.7071 pulls every delta toward observed =
    # 1.0, flipping -0.2 to +0.1515 and reporting 4 of 5.
    deltas = [-3.0, -0.2, 0.5, 2.0, 4.0]
    b = [0.0] * 5
    d = paired_delta(deltas, b, 1.0, 0.0, m=8, n=16)
    assert d.prob_positive == 0.6
    assert 1.0 + (8 / 16) ** 0.5 * (-0.2 - 1.0) > 0     # what the old factor did


@pytest.mark.parametrize(("m", "n"), [(2, None), (None, 8)])
def test_paired_delta_rejects_half_specified_draw_sizes(m, n):
    # Silently falling back to the n-out-of-n percentile path on a typo'd
    # keyword would report a subsample delta with no rescaling at all.
    with pytest.raises(ValueError, match="both m and n"):
        paired_delta([1.0, 2.0], [0.0, 1.0], 1.5, 0.5, m=m, n=n)


def test_paired_delta_propagates_the_subsample_draw_size_guard():
    with pytest.raises(ValueError, match="0 < m <= n"):
        paired_delta([1.0, 2.0], [0.0, 1.0], 1.5, 0.5, m=9, n=4)


def test_paired_delta_degenerate_when_nothing_was_resampled():
    # n == 0: every document held fixed. subsample_interval reports a zero-width
    # interval at the observed delta; prob_positive must agree with it rather
    # than report variability that does not exist.
    d = paired_delta([2.0] * 4, [1.0] * 4, 2.0, 1.0, m=0, n=0)
    assert (d.estimate, d.lo, d.hi, d.prob_positive) == (1.0, 1.0, 1.0, 1.0)
