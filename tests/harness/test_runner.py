import pytest

from lattice.config.schema import AdapterSpec
from lattice.core.types import GraphDelta
from lattice.harness.runner import ExperimentConfig, run_experiment, run_from_path
from lattice.ports import DocumentMetric
from lattice.registry.registry import register

CONFIG_PATH = "configs/walking-skeleton.toml"


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "block"},
            "extractor": {"name": "token", "params": {"min_length": 4}},
            "scorer": {"name": "frequency", "params": {"top_k": 10}},
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "dataset": {"name": "toy"},
            "metrics": [{"name": "label-f1"}],
        }
    )


def test_run_experiment_end_to_end():
    report = run_experiment(_experiment_config())
    assert report.documents_processed == 3
    assert report.errors == ()
    # every gold label is found by the trivial pipeline on the toy corpus
    assert report.metrics["label-f1"]["recall"] == 1.0
    # the trivial extractor over-generates, so precision is imperfect
    assert 0.0 < report.metrics["label-f1"]["precision"] < 1.0


def test_report_stamps_the_resolved_config():
    report = run_experiment(_experiment_config())
    assert report.config["scorer"]["name"] == "frequency"
    assert report.config["scorer"]["params"] == {"top_k": 10}
    assert report.config["run"]["seed"] == 0
    assert report.config["embedder"]["name"] == "hashing"  # default stamped too


def test_rerunning_the_same_config_reproduces_the_report():
    assert run_experiment(_experiment_config()) == run_experiment(_experiment_config())


def test_run_from_path_loads_toml_and_runs():
    report = run_from_path(CONFIG_PATH)
    assert report.documents_processed == 3
    assert report.metrics["label-f1"]["recall"] == 1.0


@register(DocumentMetric, "count-docs")
class _CountDocs(DocumentMetric):
    def evaluate_documents(self, deltas, ground_truth):
        return {"documents": float(len(list(deltas)))}


def test_document_metrics_evaluated_from_deltas():
    config = _experiment_config()
    config = config.model_copy(update={"document_metrics": [AdapterSpec(name="count-docs")]})
    report = run_experiment(config)
    assert report.metrics["count-docs"] == {"documents": 3.0}


@register(DocumentMetric, "label-f1")
class _ShadowLabelF1(DocumentMetric):
    def evaluate_documents(self, deltas, ground_truth):
        return {"x": 0.0}


def test_duplicate_metric_names_across_families_rejected():
    config = _experiment_config().model_copy(
        update={"document_metrics": [AdapterSpec(name="label-f1")]}
    )
    with pytest.raises(ValueError, match="label-f1"):
        run_experiment(config)
