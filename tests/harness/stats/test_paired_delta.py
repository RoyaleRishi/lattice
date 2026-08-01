"""The paired path, end to end: bootstrap() over two bundles -> paired_delta().

The property under test is the *pairing invariant* — iteration i of both arms
must be computed over the same replicate document set. Nothing enforces it at
runtime (paired_delta can only check equal length); it holds because
bootstrap() constructs a fresh random.Random(seed) per call and walks an
insertion-ordered pool, so two calls at the same seed over two bundles with the
same document ids in the same order consume the identical RNG sequence. These
tests assert that property directly — the same document ids per iteration, not
merely equal-length draw lists — under both schemes, because a regression there
would not fail any other test in the suite and would silently widen every
paired CI the project publishes.
"""

import pytest

from lattice.harness.stats.intervals import paired_delta
from lattice.harness.stats.records import ResampleBundle
from lattice.harness.stats.resample import bootstrap

DOC_IDS = [f"d{i}" for i in range(8)]


def _recording_bundle(scores: dict[str, float]) -> tuple[ResampleBundle, list[list[str]]]:
    """A bundle that records the exact document ids of every replicate and
    scores it as the mean of `scores`. Two bundles built from different
    `scores` over the same keys stand in for two resolver configs run on one
    corpus: same documents, different per-document values."""
    seen: list[list[str]] = []

    def aggregate(records, ctx):
        seen.append([doc_id for doc_id, _ in records])
        return {"b3-f1": sum(v for _, v in records) / len(records)}

    return (
        ResampleBundle(
            kind="pooled",
            per_document={d: (d, v) for d, v in scores.items()},
            aggregate=aggregate,
        ),
        seen,
    )


def _arm_a() -> tuple[ResampleBundle, list[list[str]]]:
    return _recording_bundle({d: 0.10 * i for i, d in enumerate(DOC_IDS)})


def _arm_b() -> tuple[ResampleBundle, list[list[str]]]:
    # deliberately different per-document values, same document ids
    return _recording_bundle({d: 0.07 * (7 - i) for i, d in enumerate(DOC_IDS)})


@pytest.mark.parametrize("scheme", ["resample", "subsample"])
def test_paired_draws_are_index_aligned_across_two_bundles(scheme):
    bundle_a, seen_a = _arm_a()
    bundle_b, seen_b = _arm_b()
    bootstrap(bundle_a, samples=50, seed=0, scheme=scheme)
    bootstrap(bundle_b, samples=50, seed=0, scheme=scheme)
    assert len(seen_a) == len(seen_b) == 50
    # the strong assertion: identical document ids, in identical order, at
    # every single iteration — not just equal-length lists
    assert seen_a == seen_b


@pytest.mark.parametrize("scheme", ["resample", "subsample"])
def test_paired_draws_are_index_aligned_with_fixed_documents(scheme):
    bundle_a, seen_a = _arm_a()
    bundle_b, seen_b = _arm_b()
    fixed = ["d0", "d1"]
    bootstrap(bundle_a, samples=30, seed=4, fixed_doc_ids=fixed, scheme=scheme)
    bootstrap(bundle_b, samples=30, seed=4, fixed_doc_ids=fixed, scheme=scheme)
    assert seen_a == seen_b
    assert all(drawn[:2] == fixed for drawn in seen_a)


@pytest.mark.parametrize("scheme", ["resample", "subsample"])
def test_alignment_is_a_property_of_the_seed_not_of_the_assertion(scheme):
    # Guards the guard: if the recording harness collapsed every draw to the
    # same thing, the alignment tests above would pass vacuously.
    bundle_a, seen_a = _arm_a()
    bundle_b, seen_b = _arm_b()
    bootstrap(bundle_a, samples=30, seed=0, scheme=scheme)
    bootstrap(bundle_b, samples=30, seed=1, scheme=scheme)
    assert seen_a != seen_b
    assert len({tuple(drawn) for drawn in seen_a}) > 1


def test_subsample_pairing_survives_a_pool_split_by_fixed_ids():
    # The fixed/pool split is computed per call. Two arms must agree on it, or
    # arm A's iteration i would be a different document set from arm B's even
    # at the same seed.
    bundle_a, seen_a = _arm_a()
    bundle_b, seen_b = _arm_b()
    bootstrap(bundle_a, samples=20, seed=2, fixed_doc_ids=["d3"], scheme="subsample")
    bootstrap(bundle_b, samples=20, seed=2, fixed_doc_ids=["d3"], scheme="subsample")
    assert seen_a == seen_b
    for drawn in seen_a:
        assert drawn[0] == "d3"
        assert len(drawn) == len(set(drawn))     # no duplicates under subsampling
        assert "d3" not in drawn[1:]             # fixed doc never redrawn


# --- the with-replacement paired delta is bit-identical for a fixed seed ------
# Captured from the implementation as it stood *before* paired_delta gained its
# m/n keywords (2026-08-01), by running the pre-change function over the draws
# these two bundles produce. The published M3 paired deltas came off this path,
# so any drift in it — in bootstrap's RNG consumption, in the delta arithmetic,
# or in the percentile convention — must fail here.

FROZEN_RESAMPLE_DELTA = (
    0.245,                    # estimate = 0.42 - 0.175
    -0.15000000000000002,     # ci lo
    0.4025,                   # ci hi
    0.746,                    # prob_positive, from the raw (unrescaled) deltas
)


def test_with_replacement_paired_delta_is_bit_identical_for_a_fixed_seed():
    bundle_a, _ = _arm_a()
    bundle_b, _ = _arm_b()
    draws_a = bootstrap(bundle_a, samples=500, seed=11)
    draws_b = bootstrap(bundle_b, samples=500, seed=11)
    assert (draws_a.scheme, draws_a.m, draws_a.n) == ("resample", 8, 8)
    d = paired_delta(draws_a.draws["b3-f1"], draws_b.draws["b3-f1"], 0.42, 0.175)
    assert (d.estimate, d.lo, d.hi, d.prob_positive) == FROZEN_RESAMPLE_DELTA


def test_subsample_paired_delta_is_seed_deterministic_and_is_the_identity_at_half():
    def run(seed: int):
        bundle_a, _ = _arm_a()
        bundle_b, _ = _arm_b()
        a = bootstrap(bundle_a, samples=500, seed=seed, scheme="subsample")
        b = bootstrap(bundle_b, samples=500, seed=seed, scheme="subsample")
        assert (a.scheme, a.m, a.n) == ("subsample", 4, 8)
        return (
            paired_delta(a.draws["b3-f1"], b.draws["b3-f1"], 0.42, 0.175, m=a.m, n=a.n),
            paired_delta(a.draws["b3-f1"], b.draws["b3-f1"], 0.42, 0.175),
        )

    sub, raw = run(11)
    assert sub == run(11)[0]              # GC3: same seed, byte-identical result
    assert sub != run(12)[0]              # and the seed is actually doing work
    assert sub.estimate == raw.estimate   # both anchored on the same observed delta
    # m = 4, n = 8 is the harness's m = n / 2, where tau = sqrt(4 / (8 - 4)) is
    # exactly 1 and the rescaling is the identity — the subsample band is the
    # plain percentile band over the paired size-4 deltas. That is the content
    # of the 10c correction: at this draw size the size-m deltas already have
    # the right spread for the full-sample delta, and the pre-2026-08-01
    # sqrt(4 / 8) = 0.707 shrank them 29% past it.
    assert sub.lo == pytest.approx(raw.lo, abs=1e-15)
    assert sub.hi == pytest.approx(raw.hi, abs=1e-15)


def test_subsample_paired_delta_rescales_when_tau_is_not_one():
    # The test above is the identity case, so on its own it would pass under
    # any factor that equals 1 at m = n / 2. This one exercises the rescaling
    # where it actually rescales, at a draw size the fixture can genuinely
    # produce: holding one document fixed leaves an odd pool of n = 7, and
    # subsample_size(7) = 4, so tau = sqrt(4 / (7 - 4)) = sqrt(4/3) = 1.1547.
    #
    # Note the direction. tau > 1 here, so the paired band is WIDER than
    # reading the same deltas as full-size bootstrap draws — the opposite of
    # what the pre-2026-08-01 sqrt(4 / 7) = 0.756 did. Odd pools are the
    # common case in this project, so this is not an exotic corner.
    bundle_a, _ = _arm_a()
    bundle_b, _ = _arm_b()
    fixed = ["d0"]
    a = bootstrap(bundle_a, samples=500, seed=13, fixed_doc_ids=fixed, scheme="subsample")
    b = bootstrap(bundle_b, samples=500, seed=13, fixed_doc_ids=fixed, scheme="subsample")
    assert (a.scheme, a.m, a.n) == ("subsample", 4, 7)

    sub = paired_delta(a.draws["b3-f1"], b.draws["b3-f1"], 0.42, 0.175, m=a.m, n=a.n)
    raw = paired_delta(a.draws["b3-f1"], b.draws["b3-f1"], 0.42, 0.175)
    tau = (4 / 3) ** 0.5

    # Widened by exactly tau, not merely "wider".
    assert (sub.hi - sub.lo) / (raw.hi - raw.lo) == pytest.approx(tau)

    # But NOT a superset of the raw band, and the reason is worth stating: the
    # rescaling is anchored on the observed delta (0.245), which on this
    # fixture lies outside the raw quantile span [-0.116, 0.224] altogether.
    # So the band scales about a point it does not contain, and moves as well
    # as widens. Asserting containment here would be asserting a coincidence.
    assert sub.lo < raw.lo and sub.hi < raw.hi
    observed = 0.42 - 0.175
    assert sub.lo == pytest.approx(observed + tau * (raw.lo - observed))
    assert sub.hi == pytest.approx(observed + tau * (raw.hi - observed))
