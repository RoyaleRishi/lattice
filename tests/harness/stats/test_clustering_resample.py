import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.core.types import GraphSnapshot
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.records import EvaluationContext, ResampleBundle
from lattice.harness.stats.resample import bootstrap

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gold-mentions",
        "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"},
    },
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "embedding-nn", "params": {"threshold": 0.8}},
    "relation_inducer": {"name": "co-occurrence"},
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "mention-clusters",
        "params": {"root": "tests/fixtures/mini_clusters_conel", "split": "test"},
    },
    "document_metrics": [{"name": "clustering"}],
})


def test_clustering_equivalence():
    report, bundles = run_experiment_detailed(CFG)
    bundle = bundles["clustering"]
    assert bundle.kind == "pooled"
    doc_ids = list(bundle.per_document)
    records = [bundle.per_document[d] for d in doc_ids]
    assert bundle.aggregate(records, bundle.global_context) == report.metrics["clustering"]


def test_emit_records_rejects_missing_ground_truth_key():
    # emit_records must guard its ground_truth["clusters_by_mention"] lookup the
    # same way evaluate_documents does — a bare subscript would raise an opaque
    # KeyError deep in the loop instead of this clear contract error.
    ctx = EvaluationContext(
        deltas=(), snapshot=GraphSnapshot(concepts=(), relations=()), ground_truth={}
    )
    with pytest.raises(ValueError, match="clusters_by_mention"):
        ClusteringMetric().emit_records(ctx)


def test_duplicating_a_document_must_not_improve_a_cross_document_error():
    """B³ is a degree-2 functional: a mention's score depends on which other
    mentions share its cluster. So a duplicated document is not a second
    observation — its two instances carry the same predicted concept id and the
    same gold cluster id and therefore agree with each other by construction,
    erasing any cross-document error the document took part in.

    This test previously asserted the opposite ("duplicating A must reweight
    the b3-precision mean toward A") on a fixture where doc A was a
    perfectly-clustered singleton — the one shape for which duplication is
    harmless — and so never noticed. The correct requirement is on the
    *estimator*: the pooled bootstrap must never hand _aggregate a duplicated
    document. Both halves below are needed; the second is what the first is
    protecting against.
    """
    # doc A and doc B each hold one mention, in separate gold clusters, wrongly
    # merged into one predicted concept: b3-precision 1/2, ARI 0.
    doc_a = [("A:0-1", "MERGED", "G1")]
    doc_b = [("B:0-1", "MERGED", "G2")]
    bundle = ResampleBundle(
        kind="pooled",
        per_document={"A": doc_a, "B": doc_b, "C": [("C:0-1", "MERGED", "G3")],
                      "D": [("D:0-1", "MERGED", "G4")]},
        aggregate=ClusteringMetric._aggregate,
    )

    # 1. The estimator the report uses for pooled bundles never duplicates, so
    #    the error is never improved away: no replicate beats the honest
    #    two-document score of 1/2 precision, and none reaches a perfect ARI.
    drawn = bootstrap(bundle, samples=200, seed=0, scheme="subsample")
    assert drawn.scheme == "subsample" and (drawn.m, drawn.n) == (2, 4)
    assert set(drawn.draws["b3-precision"]) == {0.5}
    assert set(drawn.draws["ari"]) == {0.0}

    # 2. The degree-2 failure itself, pinned so it cannot creep back in: hand
    #    _aggregate a duplicate and the cross-document error vanishes outright.
    base = ClusteringMetric._aggregate([doc_a, doc_b], {})
    dup = ClusteringMetric._aggregate([doc_a, doc_a], {})
    assert base["b3-precision"] == pytest.approx(0.5) and base["ari"] == 0.0
    assert dup["b3-precision"] == 1.0 and dup["ari"] == 1.0
