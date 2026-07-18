from lattice.harness.runner import ExperimentConfig, run_experiment_detailed

CFG = ExperimentConfig.model_validate({
    "segmenter": {"name": "block"},
    "extractor": {
        "name": "gazetteer",
        "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"},
    },
    "scorer": {"name": "passthrough"},
    "resolver": {"name": "exact-label"},
    "relation_inducer": {
        "name": "union",
        "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
    },
    "graph_integrator": {"name": "in-memory"},
    "embedder": {"name": "hashing"},
    "dataset": {
        "name": "taxonomy",
        "params": {"root": "tests/fixtures/mini_texeval", "gold": "toy"},
    },
    "metrics": [{"name": "edge-f1"}],
})


def test_edge_f1_equivalence_and_bundle_shape():
    report, bundles = run_experiment_detailed(CFG)
    assert set(bundles) == {"edge-f1"}
    bundle = bundles["edge-f1"]
    assert bundle.kind == "pooled"
    doc_ids = list(bundle.per_document)
    records = [bundle.per_document[d] for d in doc_ids]
    assert bundle.aggregate(records, bundle.global_context) == report.metrics["edge-f1"]
    # union of per-doc predicted edges equals the recomputed predicted_edges count
    union = set().union(*records) if records else set()
    assert float(len(union)) == report.metrics["edge-f1"]["predicted_edges"]
