"""Declarative sweeps (M2 spec §7): a base experiment config plus axes of
adapter alternatives, expanded as a cartesian product. Each config runs with
a fresh factory-built orchestrator — no state is reused across runs."""

import dataclasses
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lattice.config.schema import AdapterSpec
from lattice.harness.runner import ExperimentConfig, RunReport, run_experiment


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: ExperimentConfig
    axes: dict[str, list[AdapterSpec]] = Field(default_factory=dict)


def expand(sweep: SweepConfig) -> list[ExperimentConfig]:
    for axis in sweep.axes:
        if axis not in ExperimentConfig.model_fields:
            raise ValueError(f"unknown sweep axis {axis!r} (not an ExperimentConfig field)")
    axis_names = sorted(sweep.axes)
    configs: list[ExperimentConfig] = []
    for combo in itertools.product(*(sweep.axes[name] for name in axis_names)):
        data = sweep.base.model_dump()
        for name, spec in zip(axis_names, combo):
            data[name] = spec.model_dump()
        configs.append(ExperimentConfig.model_validate(data))
    return configs


@dataclass(frozen=True)
class SweepReport:
    sweep: dict[str, Any]
    runs: list[RunReport]
    table: list[dict[str, object]]


def _row(config: ExperimentConfig, report: RunReport, axis_names: list[str]) -> dict[str, object]:
    row: dict[str, object] = {
        f"axis:{name}": getattr(config, name).name for name in axis_names
    }
    for metric_name, values in sorted(report.metrics.items()):
        for key, value in sorted(values.items()):
            row[f"{metric_name}.{key}"] = value
    row["errors"] = len(report.errors)
    return row


def run_sweep(sweep: SweepConfig) -> SweepReport:
    axis_names = sorted(sweep.axes)
    configs = expand(sweep)
    runs = [run_experiment(config) for config in configs]
    table = [_row(config, report, axis_names) for config, report in zip(configs, runs)]
    return SweepReport(sweep=sweep.model_dump(), runs=runs, table=table)


def _markdown_table(table: list[dict[str, object]]) -> str:
    if not table:
        return "(empty sweep)\n"
    columns: list[str] = []
    for row in table:
        for key in row:
            if key not in columns:
                columns.append(key)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in table:
        lines.append(
            "| "
            + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in (row.get(c, "") for c in columns)
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_reports(report: SweepReport, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "sweep-report.json"
    md_path = out / "sweep-report.md"
    json_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True))
    md_path.write_text(_markdown_table(report.table))
    return json_path, md_path
