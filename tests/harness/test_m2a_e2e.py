import pytest

from lattice.harness.runner import ExperimentConfig, run_experiment

MINI_ROOT = "tests/fixtures/mini_inspec"


def _config(extractor: dict, embedder: dict, scorer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": extractor,
            "scorer": scorer,
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": embedder,
            "dataset": {"name": "inspec", "params": {"root": MINI_ROOT}},
            "document_metrics": [{"name": "f1-at-k"}],
        }
    )


def test_document_metric_pipeline_pure():
    """The full per-document evaluation path with M1's pure adapters — proves
    the M2a wiring without the ml stack."""
    report = run_experiment(
        _config(
            extractor={"name": "token", "params": {"min_length": 4}},
            embedder={"name": "hashing"},
            scorer={"name": "frequency", "params": {"top_k": 15}},
        )
    )
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["f1-at-k"]) == {
        "precision@5", "recall@5", "f1@5",
        "precision@10", "recall@10", "f1@10",
        "precision@15", "recall@15", "f1@15",
    }


@pytest.mark.ml
def test_real_baseline_on_mini_fixture():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(
            _config(
                extractor={"name": "noun-chunk"},
                embedder={"name": "sentence-transformer"},
                scorer={"name": "embedding-cosine", "params": {"top_k": 15}},
            )
        )
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert report.metrics["f1-at-k"]["recall@10"] > 0.4  # gold phrases appear verbatim
    assert report.config["embedder"]["name"] == "sentence-transformer"


@pytest.mark.ml
def test_real_baseline_is_reproducible():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    config = _config(
        extractor={"name": "noun-chunk"},
        embedder={"name": "sentence-transformer"},
        scorer={"name": "embedding-cosine", "params": {"top_k": 15}},
    )
    try:
        assert run_experiment(config) == run_experiment(config)
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
