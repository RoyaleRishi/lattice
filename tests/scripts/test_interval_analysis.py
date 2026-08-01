"""Orchestration tests for scripts/interval_analysis.py.

The real script runs sentence-transformers over ConEL-2/ECB+, so the pipeline
is stubbed out: `_clustering_bundle` is replaced by synthetic bundles over the
same document ids. What is exercised is the orchestration the script actually
owns — which arms it builds, which pairs it emits, which resampling scheme it
passes, and whether it fails loudly when the pairing precondition is violated.
Config loading and the in-memory resolver swap are exercised for real.
"""

import pytest

from lattice.harness.stats.records import ResampleBundle
from scripts import interval_analysis as ia

DOC_IDS = [f"d{i}" for i in range(8)]


def _bundle(offset: float) -> ResampleBundle:
    return ResampleBundle(
        kind="pooled",
        per_document={d: 0.5 + offset * i for i, d in enumerate(DOC_IDS)},
        aggregate=lambda recs, ctx: {"b3-f1": sum(recs) / len(recs)},
    )


@pytest.fixture
def fast(monkeypatch):
    """Cheap stand-ins for the pipeline, keyed by resolver name so each arm
    gets a distinguishable score."""
    offsets = {"exact-label": 0.01, "stemmed-label": 0.02, "embedding-nn": 0.03}

    def fake_clustering_bundle(config):
        bundle = _bundle(offsets.get(config.resolver.name, 0.04))
        estimate = bundle.aggregate(list(bundle.per_document.values()), {})["b3-f1"]
        return estimate, bundle

    monkeypatch.setattr(ia, "_clustering_bundle", fake_clustering_bundle)
    monkeypatch.setattr(ia, "ITEM_SAMPLES", 200)
    return fake_clustering_bundle


def test_arm_configs_are_the_three_way_identity_ladder():
    # Real config load + real in-memory resolver swap, no pipeline run.
    arms = ia._m3_arm_configs("conel2")
    assert list(arms) == ["exact-label", "stemmed-label", "embedding-nn@0.90"]
    assert arms["exact-label"].resolver.name == "exact-label"
    assert arms["stemmed-label"].resolver.name == "stemmed-label"
    assert arms["embedding-nn@0.90"].resolver.name == "embedding-nn"
    assert arms["embedding-nn@0.90"].resolver.params["threshold"] == 0.90
    # the stemmed arm is the exact-label config with only the resolver changed
    exact, stemmed = arms["exact-label"].model_dump(), arms["stemmed-label"].model_dump()
    assert {k: v for k, v in exact.items() if k != "resolver"} == {
        k: v for k, v in stemmed.items() if k != "resolver"
    }


def test_with_threshold_only_replaces_the_resolver():
    base = ia._load(ia.M3_CONFIGS["ecbplus"]["exact-label"])
    variant = ia._with_threshold(base, 0.75)
    assert variant.resolver.name == "embedding-nn"
    assert variant.resolver.params == {"threshold": 0.75}
    assert variant.dataset == base.dataset and variant.extractor == base.extractor


def test_paired_deltas_cover_all_three_pairs_and_use_the_subsample_scheme(fast):
    rows = ia.m3_paired_deltas("conel2")
    assert [(r["arm_a"], r["arm_b"]) for r in rows] == list(ia.M3_PAIRS)
    for row in rows:
        assert row["scheme"] == "subsample"
        assert (row["m"], row["n"]) == (4, 8)     # subsample_size(8) == 4
        assert row["pair"] == f"{row['arm_a']} - {row['arm_b']}"
        assert row["delta_estimate"] == pytest.approx(row["b3_f1_a"] - row["b3_f1_b"])
        assert row["ci_lo"] <= row["ci_hi"]


def test_paired_deltas_are_internally_consistent_across_the_ladder(fast):
    # (nn - exact) must equal (nn - stemmed) + (stemmed - exact) on the point
    # estimates; a mis-wired pair table would break this.
    by_pair = {(r["arm_a"], r["arm_b"]): r["delta_estimate"] for r in ia.m3_paired_deltas("conel2")}
    assert by_pair[(ia.NN090, ia.EXACT)] == pytest.approx(
        by_pair[(ia.NN090, ia.STEMMED)] + by_pair[(ia.STEMMED, ia.EXACT)]
    )


def test_paired_deltas_reject_arms_that_enumerate_documents_differently(monkeypatch):
    # The pairing invariant is "same seed over the same insertion-ordered
    # pool". If one arm's bundle held different (or reordered) documents, the
    # draws would not be paired and the delta CI would be silently wrong.
    def skewed(config):
        bundle = (
            _bundle(0.01)
            if config.resolver.name == "exact-label"
            else ResampleBundle(
                kind="pooled",
                per_document={d: 0.5 for d in reversed(DOC_IDS)},
                aggregate=lambda recs, ctx: {"b3-f1": sum(recs) / len(recs)},
            )
        )
        return 0.5, bundle

    monkeypatch.setattr(ia, "_clustering_bundle", skewed)
    monkeypatch.setattr(ia, "ITEM_SAMPLES", 20)
    with pytest.raises(ValueError, match="would not be paired"):
        ia.m3_paired_deltas("conel2")


def test_threshold_curve_uses_the_subsample_interval_not_bca(fast):
    rows = ia.m3_threshold_curve("conel2")
    assert [r["threshold"] for r in rows] == ia.THRESHOLD_GRID
    assert sum(r["is_operating_point"] for r in rows) == 1
    for row in rows:
        assert row["ci_method"] == "subsample"
        assert row["scheme"] == "subsample"
        assert (row["m"], row["n"]) == (4, 8)
        assert row["brackets_estimate"] == (row["ci_lo"] <= row["b3_f1"] <= row["ci_hi"])
