import pytest

from lattice.config.factory import instantiate
from lattice.harness.runner import (
    ExperimentConfig,
    run_experiment,
    run_experiment_detailed,
    run_on_documents,
)
from lattice.ports import Dataset

M5 = ExperimentConfig.model_validate({
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
    "metrics": [{"name": "redundancy"}],
    "document_metrics": [{"name": "clustering"}],
})


def test_run_on_documents_matches_run_experiment_on_full_stream():
    docs = list(instantiate(Dataset, M5.dataset).documents())
    flat = run_on_documents(M5, docs)
    report = run_experiment(M5)
    expected = {
        f"{name}.{k}": v
        for name, values in report.metrics.items()
        for k, v in values.items()
    }
    assert flat == expected


def test_run_on_documents_rejects_name_in_both_families():
    from lattice.config.factory import instantiate
    from lattice.ports import Dataset
    data = M5.model_dump()
    data["metrics"] = [{"name": "clustering"}]
    data["document_metrics"] = [{"name": "clustering"}]
    config = ExperimentConfig.model_validate(data)
    docs = list(instantiate(Dataset, config.dataset).documents())
    with pytest.raises(ValueError, match="both families"):
        run_on_documents(config, docs)


def test_run_experiment_detailed_rejects_name_in_both_families():
    data = M5.model_dump()
    data["metrics"] = [{"name": "clustering"}]
    data["document_metrics"] = [{"name": "clustering"}]
    config = ExperimentConfig.model_validate(data)
    with pytest.raises(ValueError, match="both families"):
        run_experiment_detailed(config)


# A test-only resamplable metric, registered at MODULE level (import-time, once):
# registry.register raises on duplicate names, so never register inside a test body.
from lattice.harness.stats.records import Resamplable, ResampleBundle  # noqa: E402
from lattice.ports import DocumentMetric  # noqa: E402
from lattice.registry.registry import register  # noqa: E402


@register(DocumentMetric, "toy-macro")
class ToyMacro(DocumentMetric, Resamplable):
    kind = "macro"

    def evaluate_documents(self, deltas, ground_truth):
        return {"n": float(len(list(deltas)))}

    def emit_records(self, context):
        return ResampleBundle(
            kind="macro",
            per_document={d.document_id: {"n": 1.0} for d in context.deltas},
            aggregate=lambda records, ctx: {"n": sum(r["n"] for r in records)},
        )


def test_detailed_returns_bundles_for_resamplable_only():
    # metrics=[] so only the resamplable toy-macro document_metric yields a bundle.
    data = M5.model_dump()
    data.update({"document_metrics": [{"name": "toy-macro"}], "metrics": []})
    config = ExperimentConfig.model_validate(data)
    report, bundles = run_experiment_detailed(config)
    assert set(bundles) == {"toy-macro"}
    assert bundles["toy-macro"].kind == "macro"
    assert set(bundles["toy-macro"].per_document)         # one record per document
