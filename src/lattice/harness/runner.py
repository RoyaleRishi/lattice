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
from lattice.ports import Dataset, Metric


class ExperimentConfig(RunConfig):
    dataset: AdapterSpec
    metrics: list[AdapterSpec] = Field(default_factory=list)


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
    metric_results = {
        spec.name: instantiate(Metric, spec).evaluate(snapshot, ground_truth)
        for spec in config.metrics
    }
    return RunReport(
        config=config.model_dump(),
        documents_processed=len(deltas),
        errors=tuple(error for delta in deltas for error in delta.errors),
        metrics=metric_results,
    )


def run_from_path(path: str | Path) -> RunReport:
    return run_experiment(load_config(path, model=ExperimentConfig))
