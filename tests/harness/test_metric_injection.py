from lattice.core.types import GraphSnapshot
from lattice.harness.runner import ExperimentConfig, run_experiment
from lattice.ports import Embedder, Metric
from lattice.registry.registry import register


@register(Metric, "test-embedder-probe")
class EmbedderProbe(Metric):
    """Test-only metric: proves the runner injects the shared embedder."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        [vector] = self.embedder.embed(["probe"])
        return {"embedder-dim": float(len(vector))}


def _config(metrics: list[dict]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token"},
            "scorer": {"name": "frequency"},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": {"name": "hashing", "params": {"dim": 16}},
            "dataset": {"name": "toy"},
            "metrics": metrics,
        }
    )


def test_metric_with_embedder_param_receives_the_configured_embedder():
    report = run_experiment(_config([{"name": "test-embedder-probe"}]))
    assert report.errors == ()
    # dim=16 proves the injected instance was built from config.embedder,
    # not a default.
    assert report.metrics["test-embedder-probe"]["embedder-dim"] == 16.0


def test_metric_without_embedder_param_is_unaffected():
    report = run_experiment(_config([{"name": "label-f1"}]))
    assert set(report.metrics["label-f1"]) == {"precision", "recall", "f1"}
