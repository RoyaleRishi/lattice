from lattice.harness.runner import ExperimentConfig

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gold-mentions",
        "params": {
            "root": "tests/fixtures/mini_clusters_conel",
            "split": "test",
        },
    },
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
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
    "document_metrics": [{"name": "clustering"}],
})


def test_f1_at_k_equivalence_via_unit_deltas():
    # Exercise emit_records/_aggregate directly on hand-made deltas
    # (like tests/helpers.make_delta).
    from lattice.adapters.document_metric.f1_at_k import F1AtK
    from lattice.core.types import GraphSnapshot
    from lattice.harness.stats.records import EvaluationContext
    from tests.helpers import make_delta

    gt = {"keyphrases_by_document": {"d1": ["alpha"], "d2": ["beta"]}}
    deltas = (make_delta("d1", [("alpha", 1.0)]), make_delta("d2", [("gamma", 1.0)]))
    metric = F1AtK(ks=[5])
    assert metric.kind == "macro"
    direct = metric.evaluate_documents(deltas, gt)
    bundle = metric.emit_records(EvaluationContext(deltas, GraphSnapshot((), ()), gt))
    records = [bundle.per_document[d.document_id] for d in deltas]
    assert bundle.aggregate(records, bundle.global_context) == direct
    assert direct["f1@5"] == 0.5
