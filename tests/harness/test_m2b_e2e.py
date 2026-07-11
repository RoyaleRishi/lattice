import pytest

from lattice.harness.runner import ExperimentConfig, run_experiment

MINI_ROOT = "tests/fixtures/mini_inspec"

METRIC_KEYS = {
    "precision@5", "recall@5", "f1@5",
    "precision@10", "recall@10", "f1@10",
    "precision@15", "recall@15", "f1@15",
}


def _config(extractor: dict, embedder: dict, scorer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "sentence"},
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


def _pure(scorer: str) -> ExperimentConfig:
    return _config(
        extractor={"name": "token", "params": {"min_length": 4}},
        embedder={"name": "hashing"},
        scorer={"name": scorer, "params": {"top_k": 15}},
    )


def _ml(scorer: str) -> ExperimentConfig:
    return _config(
        extractor={"name": "noun-chunk"},
        embedder={"name": "sentence-transformer"},
        scorer={"name": scorer, "params": {"top_k": 15}},
    )


@pytest.mark.parametrize("scorer", ["mderank", "hcuke"])
def test_m2b_scorer_pipeline_pure(scorer):
    """Both frontier scorers through the full harness with M1's pure adapters:
    proves registration and wiring without the ml stack."""
    report = run_experiment(_pure(scorer))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["f1-at-k"]) == METRIC_KEYS


@pytest.mark.ml
@pytest.mark.parametrize("scorer", ["mderank", "hcuke"])
def test_m2b_scorer_real_ml_path(scorer):
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(_ml(scorer))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    # Quality thresholds live on the real Inspec benchmark (M2a lesson: a
    # 3-document fixture cannot hold one); this proves the real ML path runs
    # end-to-end with defined metrics.
    assert 0.0 <= report.metrics["f1-at-k"]["f1@15"] <= 1.0


@pytest.mark.ml
def test_hcuke_is_reproducible():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        assert run_experiment(_ml("hcuke")) == run_experiment(_ml("hcuke"))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
