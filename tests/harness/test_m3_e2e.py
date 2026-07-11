import pytest

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.harness.sweep import SweepConfig, expand

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"
CONEL_ROOT = "tests/fixtures/mini_clusters_conel"
METRIC_KEYS = {"b3-precision", "b3-recall", "b3-f1", "ari"}


def _config(root: str, resolver: dict, embedder: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gold-mentions", "params": {"root": root}},
            "scorer": {"name": "passthrough"},
            "resolver": resolver,
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": embedder,
            "dataset": {"name": "mention-clusters", "params": {"root": root}},
            "document_metrics": [{"name": "clustering"}],
        }
    )


@pytest.mark.parametrize("root", [ECB_ROOT, CONEL_ROOT])
@pytest.mark.parametrize(
    "resolver",
    [{"name": "exact-label"}, {"name": "embedding-nn", "params": {"threshold": 0.8}}],
)
def test_m3_pipeline_pure(root, resolver):
    """Both corpora fixtures x both resolvers through the full harness with
    the hashing embedder — proves wiring without the ml stack."""
    report = run_experiment(_config(root, resolver, {"name": "hashing"}))
    assert report.errors == ()
    assert report.documents_processed == 3
    metrics = report.metrics["clustering"]
    assert set(metrics) == METRIC_KEYS
    assert all(-1.0 <= v <= 1.0 for v in metrics.values())


def test_mismatched_roots_fail_loudly():
    # Dataset from one corpus, sidecar from the other: the gold-mentions
    # extractor must refuse, not silently under-report.
    config = ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "gold-mentions", "params": {"root": CONEL_ROOT}},
            "scorer": {"name": "passthrough"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing"},
            "dataset": {"name": "mention-clusters", "params": {"root": ECB_ROOT}},
            "document_metrics": [{"name": "clustering"}],
        }
    )
    with pytest.raises(ValueError, match="not in gold mention sidecar"):
        run_experiment(config)


def test_m3_run_is_reproducible():
    config = _config(
        ECB_ROOT, {"name": "embedding-nn", "params": {"threshold": 0.8}},
        {"name": "hashing"},
    )
    assert run_experiment(config) == run_experiment(config)


@pytest.mark.parametrize(
    "path", ["configs/m3-ecbplus-sweep.toml", "configs/m3-conel2-sweep.toml"]
)
def test_m3_sweep_configs_expand_to_six(path):
    sweep = load_config(path, model=SweepConfig)
    configs = expand(sweep)
    assert len(configs) == 6
    assert [c.resolver.name for c in configs] == [
        "exact-label"] + ["embedding-nn"] * 5


@pytest.mark.ml
def test_m3_real_embedder_path():
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(
            _config(
                CONEL_ROOT,
                {"name": "embedding-nn", "params": {"threshold": 0.8}},
                {"name": "sentence-transformer"},
            )
        )
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    assert set(report.metrics["clustering"]) == METRIC_KEYS
