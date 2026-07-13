import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ROOT = "tests/fixtures/mini_clusters_conel"
REDUNDANCY_KEYS = {"duplicate-rate", "near-duplicate-pairs", "concept-count"}
SANITY_KEYS = {
    "cycle-components", "cycle-nodes", "self-loops",
    "max-depth", "transitive-shortcuts", "is-a-edges",
}
COHERENCE_KEYS = {"coherence", "multi-surface-concepts", "singleton-fraction"}


def _config(resolver: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token"},
            "scorer": {"name": "embedding-cosine"},
            "resolver": resolver,
            "relation_inducer": {
                "name": "union",
                "params": {"members": [{"name": "hearst"}, {"name": "compound"}]},
            },
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {
                "name": "mention-clusters",
                "params": {"root": ROOT, "split": "test"},
            },
            "metrics": [{"name": "redundancy"}, {"name": "hierarchy-sanity"}],
            "document_metrics": [{"name": "coherence"}],
        }
    )


@pytest.mark.parametrize(
    "resolver",
    [
        {"name": "exact-label"},
        {"name": "embedding-nn", "params": {"threshold": 0.8}},
    ],
)
def test_m5_intrinsic_pipeline_pure(resolver):
    """Full real-shape pipeline (token extractor standing in for spaCy) with
    all three intrinsic metrics — proves wiring without the ml stack."""
    report = run_experiment(_config(resolver))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["redundancy"]) == REDUNDANCY_KEYS
    assert set(report.metrics["hierarchy-sanity"]) == SANITY_KEYS
    assert set(report.metrics["coherence"]) == COHERENCE_KEYS
    assert 0.0 <= report.metrics["redundancy"]["duplicate-rate"] <= 1.0
    assert 0.0 <= report.metrics["coherence"]["singleton-fraction"] <= 1.0
    assert report.metrics["hierarchy-sanity"]["self-loops"] == 0.0


def test_m5_run_is_reproducible():
    config = _config({"name": "embedding-nn", "params": {"threshold": 0.8}})
    assert run_experiment(config) == run_experiment(config)


def test_m5_sweep_config_expands_to_four_resolver_rows():
    sweep = load_config("configs/m5-conel2-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 4
    assert [c.resolver.name for c in configs] == [
        "exact-label", "embedding-nn", "embedding-nn", "embedding-nn",
    ]
    assert [
        c.resolver.params.get("threshold") for c in configs
    ] == [None, 0.90, 0.75, 0.65]
    for config in configs:
        assert config.extractor.name == "noun-chunk"
        assert config.embedder.name == "sentence-transformer"
        assert {m.name for m in config.metrics} == {"redundancy", "hierarchy-sanity"}
        assert [m.name for m in config.document_metrics] == ["coherence"]


@pytest.mark.ml
def test_m5_real_pipeline_path():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    config = _config({"name": "embedding-nn", "params": {"threshold": 0.8}})
    data = config.model_dump()
    data["extractor"] = {"name": "noun-chunk"}
    data["embedder"] = {"name": "sentence-transformer"}
    try:
        report = run_experiment(ExperimentConfig.model_validate(data))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert set(report.metrics["coherence"]) == COHERENCE_KEYS
