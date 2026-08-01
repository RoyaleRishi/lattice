import pytest

from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.records import ResampleBundle
from lattice.harness.stats.resample import (
    bootstrap,
    bootstrap_holistic,
    jackknife,
    subsample_size,
)


def _sum_bundle():
    return ResampleBundle(
        kind="pooled",
        per_document={"a": 1.0, "b": 2.0, "c": 3.0},
        aggregate=lambda records, ctx: {"total": float(sum(records))},
    )


def _recording_bundle_ids(doc_ids: list[str]) -> tuple[ResampleBundle, list[list[str]]]:
    """A bundle whose per-document record IS its document id, plus the list the
    aggregate appends each drawn replicate to — lets a test read the exact
    document multiset the engine drew, not just the statistic it produced."""
    seen: list[list[str]] = []

    def aggregate(records, ctx):
        seen.append(list(records))
        return {"count": float(len(records))}

    return (
        ResampleBundle(
            kind="pooled",
            per_document={d: d for d in doc_ids},
            aggregate=aggregate,
        ),
        seen,
    )


def test_bootstrap_is_seed_deterministic_and_varies():
    b = _sum_bundle()
    assert bootstrap(b, samples=200, seed=7) == bootstrap(b, samples=200, seed=7)
    assert bootstrap(b, samples=200, seed=7) != bootstrap(b, samples=200, seed=8)


def test_bootstrap_all_fixed_is_constant_point_estimate():
    b = _sum_bundle()
    out = bootstrap(b, samples=25, seed=1, fixed_doc_ids=["a", "b", "c"])
    assert out.draws["total"] == [6.0] * 25
    assert (out.scheme, out.m, out.n) == ("resample", 0, 0)


def test_bootstrap_rejects_unknown_scheme():
    # A typo'd scheme must fail loudly rather than silently fall through to the
    # with-replacement default, which is the wrong estimator for pooled metrics.
    with pytest.raises(ValueError, match="unknown scheme"):
        bootstrap(_sum_bundle(), samples=5, seed=0, scheme="jackknife")


# --- the "resample" path is frozen -------------------------------------------
# These expectations were captured from the implementation as it stood *before*
# the subsampling scheme was introduced (2026-07-31). The macro/f1-at-k path and
# every published M2/M3 interval depend on this exact draw sequence, so any
# change to the RNG consumption pattern must show up here as a hard failure.

FROZEN_DRAWS_SEED0 = [
    ["d3", "d3", "d0", "d2"],
    ["d3", "d3", "d2", "d3"],
    ["d2", "d1", "d1", "d2"],
    ["d1", "d0", "d2", "d1"],
    ["d2", "d0", "d0", "d2"],
    ["d3", "d0", "d2", "d3"],
]
FROZEN_DRAWS_SEED0_FIXED_D0 = [
    ["d0", "d2", "d2", "d1"],
    ["d0", "d2", "d3", "d2"],
    ["d0", "d2", "d2", "d2"],
    ["d0", "d2", "d3", "d1"],
]
FROZEN_MEAN_SEED1234 = [2.4, 1.6, 2.2, 3.2, 2.6, 2.6, 3.4, 3.2]
FROZEN_MAX_SEED1234 = [5.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


def test_resample_scheme_draw_sequence_is_frozen():
    bundle, seen = _recording_bundle_ids(["d0", "d1", "d2", "d3"])
    bootstrap(bundle, samples=6, seed=0)
    assert seen == FROZEN_DRAWS_SEED0


def test_resample_scheme_draw_sequence_is_frozen_with_fixed_ids():
    bundle, seen = _recording_bundle_ids(["d0", "d1", "d2", "d3"])
    bootstrap(bundle, samples=4, seed=0, fixed_doc_ids=["d0"])
    assert seen == FROZEN_DRAWS_SEED0_FIXED_D0


def test_resample_scheme_values_are_frozen():
    bundle = ResampleBundle(
        kind="macro",
        per_document={"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0},
        aggregate=lambda recs, ctx: {"mean": sum(recs) / len(recs), "max": max(recs)},
    )
    out = bootstrap(bundle, samples=8, seed=1234)
    assert out.scheme == "resample" and out.m == 5 and out.n == 5
    assert out.draws["mean"] == FROZEN_MEAN_SEED1234
    assert out.draws["max"] == FROZEN_MAX_SEED1234


# --- the "subsample" path ------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 3), (7, 4), (9, 4), (10, 5)],
)
def test_subsample_size_halves_floored_at_two_and_capped_at_n(n, expected):
    # floor of 2 keeps a pair in every replicate (B³/ARI are pairwise); the cap
    # keeps a pool below the floor drawable at all. n=5 -> 2 and n=9 -> 4 are
    # round()'s banker's rounding on the .5 cases, which is fine: it only has to
    # be deterministic (GC3).
    assert subsample_size(n) == expected


def test_subsample_draws_have_no_duplicate_documents():
    # The whole point of the scheme: a document may not appear twice in one
    # replicate, because _aggregate re-keys instances and two copies of a
    # document agree with themselves in both the predicted and gold partition.
    bundle, seen = _recording_bundle_ids([f"d{i}" for i in range(9)])
    out = bootstrap(bundle, samples=50, seed=3, scheme="subsample")
    assert (out.scheme, out.m, out.n) == ("subsample", 4, 9)
    assert seen and all(len(drawn) == len(set(drawn)) == 4 for drawn in seen)


def test_subsample_keeps_fixed_documents_out_of_the_pool():
    bundle, seen = _recording_bundle_ids([f"d{i}" for i in range(9)])
    out = bootstrap(
        bundle, samples=30, seed=3, fixed_doc_ids=["d0", "d1"], scheme="subsample"
    )
    assert (out.m, out.n) == (4, 7)     # pool is d2..d8; m = round(7/2) = 4
    for drawn in seen:
        assert drawn[:2] == ["d0", "d1"]           # fixed, always, in order
        assert len(drawn) == len(set(drawn)) == 6  # 2 fixed + 4 drawn, all distinct
        assert not {"d0", "d1"} & set(drawn[2:])   # never redrawn from the pool


def test_subsample_is_seed_deterministic():
    b = _sum_bundle()
    a1 = bootstrap(b, samples=100, seed=7, scheme="subsample")
    a2 = bootstrap(b, samples=100, seed=7, scheme="subsample")
    assert a1 == a2
    assert a1 != bootstrap(b, samples=100, seed=8, scheme="subsample")


def test_subsample_differs_from_resample_on_the_same_seed():
    # Guards against the scheme argument being accepted and then ignored.
    b = _sum_bundle()
    assert (
        bootstrap(b, samples=100, seed=7, scheme="subsample").draws
        != bootstrap(b, samples=100, seed=7, scheme="resample").draws
    )


def test_jackknife_leaves_one_out():
    b = _sum_bundle()
    jk = jackknife(b)
    assert sorted(jk["total"]) == [3.0, 4.0, 5.0]     # sum minus each of a,b,c


HCFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gold-mentions",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "scorer": {"name": "passthrough"},
    "resolver": {
        "name": "embedding-nn",
        "params": {"threshold": 0.8},
    },
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "mention-clusters",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "metrics": [{"name": "redundancy"}],
})


def test_bootstrap_holistic_runs_and_is_deterministic():
    a = bootstrap_holistic(HCFG, samples=5, seed=3)
    assert a == bootstrap_holistic(HCFG, samples=5, seed=3)     # same seed -> identical
    assert a != bootstrap_holistic(HCFG, samples=5, seed=4)     # different seed -> differs
    assert "redundancy.concept-count" in a
    assert len(a["redundancy.concept-count"]) == 5
