"""Experiment runner (spec §9): load dataset → fold process() over its
documents → snapshot the graph → score with metrics → emit a report stamped
with the fully resolved config (spec §7 reproducibility)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.loader import load_config
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.ports import Dataset, DocumentMetric, Embedder, Metric


class ExperimentConfig(RunConfig):
    dataset: AdapterSpec
    metrics: list[AdapterSpec] = Field(default_factory=list)
    document_metrics: list[AdapterSpec] = Field(default_factory=list)


@dataclass(frozen=True)
class RunReport:
    config: dict[str, Any]
    documents_processed: int
    errors: tuple[str, ...]
    metrics: dict[str, dict[str, float]]


def run_experiment(config: ExperimentConfig) -> RunReport:
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    deltas = orchestrator.process_stream(dataset.documents())
    snapshot = orchestrator.snapshot()
    ground_truth = dataset.ground_truth()
    # Intrinsic metrics may consume the embedder (M5 spec §3): a second
    # instance from the same spec is deterministic, so pipeline and metric
    # embeddings agree; instantiate() injects only where the constructor
    # names the param.
    metric_shared: dict[str, object] = {
        "embedder": instantiate(Embedder, config.embedder)
    }
    metric_results = {
        spec.name: instantiate(Metric, spec, metric_shared).evaluate(
            snapshot, ground_truth
        )
        for spec in config.metrics
    }
    document_results = {
        spec.name: instantiate(DocumentMetric, spec, metric_shared).evaluate_documents(
            deltas, ground_truth
        )
        for spec in config.document_metrics
    }
    duplicates = set(metric_results) & set(document_results)
    if duplicates:
        raise ValueError(f"metric name(s) used by both families: {sorted(duplicates)}")
    all_metrics = {**metric_results, **document_results}
    return RunReport(
        config=config.model_dump(),
        documents_processed=len(deltas),
        errors=tuple(error for delta in deltas for error in delta.errors),
        metrics=all_metrics,
    )


def run_from_path(path: str | Path) -> RunReport:
    return run_experiment(load_config(path, model=ExperimentConfig))
