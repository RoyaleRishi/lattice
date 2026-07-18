"""Experiment runner (spec §9): load dataset → fold process() over its
documents → snapshot the graph → score with metrics → emit a report stamped
with the fully resolved config (spec §7 reproducibility)."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.loader import load_config
from lattice.config.schema import AdapterSpec, RunConfig
from lattice.core.types import Document
from lattice.harness.stats.records import EvaluationContext, Resamplable, ResampleBundle
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


def run_on_documents(
    config: ExperimentConfig, documents: Sequence[Document]
) -> dict[str, float]:
    """Run the pipeline over an explicit document list (a resample or a
    permutation) and return a flat {"<metric>.<key>": value} dict. Mirrors
    run_experiment's scoring but with a caller-supplied stream."""
    orchestrator = build_orchestrator(config)
    deltas = orchestrator.process_stream(documents)
    snapshot = orchestrator.snapshot()
    dataset = instantiate(Dataset, config.dataset)
    ground_truth = dataset.ground_truth()
    metric_shared: dict[str, object] = {"embedder": instantiate(Embedder, config.embedder)}
    flat: dict[str, float] = {}
    for spec in config.metrics:
        for key, value in instantiate(Metric, spec, metric_shared).evaluate(
            snapshot, ground_truth
        ).items():
            flat[f"{spec.name}.{key}"] = value
    for spec in config.document_metrics:
        for key, value in instantiate(DocumentMetric, spec, metric_shared).evaluate_documents(
            deltas, ground_truth
        ).items():
            flat[f"{spec.name}.{key}"] = value
    return flat


def run_experiment_detailed(
    config: ExperimentConfig,
) -> tuple[RunReport, dict[str, ResampleBundle]]:
    """Run the experiment once and, for every macro/pooled Resamplable metric,
    capture a ResampleBundle of per-document detail for item-level bootstrap.
    Holistic and non-resamplable metrics contribute no bundle."""
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    deltas = orchestrator.process_stream(dataset.documents())
    snapshot = orchestrator.snapshot()
    ground_truth = dataset.ground_truth()
    metric_shared: dict[str, object] = {"embedder": instantiate(Embedder, config.embedder)}
    context = EvaluationContext(tuple(deltas), snapshot, ground_truth)
    metric_results: dict[str, dict[str, float]] = {}
    document_results: dict[str, dict[str, float]] = {}
    bundles: dict[str, ResampleBundle] = {}
    for spec in config.metrics:
        metric = instantiate(Metric, spec, metric_shared)
        metric_results[spec.name] = metric.evaluate(snapshot, ground_truth)
        if isinstance(metric, Resamplable) and metric.kind in ("macro", "pooled"):
            bundles[spec.name] = metric.emit_records(context)
    for spec in config.document_metrics:
        metric = instantiate(DocumentMetric, spec, metric_shared)
        document_results[spec.name] = metric.evaluate_documents(deltas, ground_truth)
        if isinstance(metric, Resamplable) and metric.kind in ("macro", "pooled"):
            bundles[spec.name] = metric.emit_records(context)
    report = RunReport(
        config=config.model_dump(),
        documents_processed=len(deltas),
        errors=tuple(error for delta in deltas for error in delta.errors),
        metrics={**metric_results, **document_results},
    )
    return report, bundles


def run_from_path(path: str | Path) -> RunReport:
    return run_experiment(load_config(path, model=ExperimentConfig))
