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


def test_clustering_equivalence_and_multiplicity():
    report, bundles = run_experiment_detailed(CFG)
    bundle = bundles["clustering"]
    assert bundle.kind == "pooled"
    doc_ids = list(bundle.per_document)
    records = [bundle.per_document[d] for d in doc_ids]
    assert bundle.aggregate(records, bundle.global_context) == report.metrics["clustering"]
    # duplicating one document must not collapse or crash; keys stay the same set
    dup = bundle.aggregate(records + [records[0]], bundle.global_context)
    assert set(dup) == set(report.metrics["clustering"])
