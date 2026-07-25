from lattice.harness.stats.intervals import (
    bca_interval,
    paired_delta,
    percentile_interval,
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


def test_paired_delta_sign_and_probability():
    d = paired_delta([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], 2.0, 3.0)
    assert d.estimate == -1.0
    assert d.prob_positive == 0.0
    assert d.lo == -1.0 and d.hi == -1.0
