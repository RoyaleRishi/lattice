import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed

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


def test_clustering_aggregate_respects_multiplicity():
    # doc A: one perfect-precision mention; doc B: two mentions sharing a predicted
    # cluster split across two gold clusters (precision 1/2 each). Duplicating A must
    # reweight the b3-precision mean toward A — only happens if _aggregate re-keys each
    # document instance uniquely. A collapsed (non-prefixed) impl gives the same value
    # for [A,B] and [A,A,B], so this fails iff the index-prefixing is removed.
    doc_a = [("A:0-1", "C1", "G1")]
    doc_b = [("B:0-1", "C2", "G2"), ("B:2-3", "C2", "G3")]
    base = ClusteringMetric._aggregate([doc_a, doc_b], {})
    dup = ClusteringMetric._aggregate([doc_a, doc_a, doc_b], {})
    assert base["b3-precision"] == pytest.approx(2 / 3)   # (1 + 1/2 + 1/2) / 3
    assert dup["b3-precision"] == pytest.approx(3 / 4)     # (1 + 1 + 1/2 + 1/2) / 4
    assert dup["b3-precision"] != base["b3-precision"]
