"""Assemble the interval report from bundles + engine + intervals, and write
it as regenerable JSON. Item-level metrics bootstrap from one pipeline run;
holistic metrics re-run the pipeline per resample."""

import json
from collections.abc import Sequence
from pathlib import Path

from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import bca_interval, percentile_interval
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife


def _iv(estimate: float, resamples: list[float], jack: list[float], level: float) -> dict:
    bca = bca_interval(estimate, resamples, jack, level)
    pct = percentile_interval(estimate, resamples, level)
    return {
        "estimate": estimate,
        "bca": {"lo": bca.lo, "hi": bca.hi, "method": bca.method},
        "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
    }


def analyze(
    config: ExperimentConfig,
    *,
    samples: int,
    seed: int,
    level: float = 0.95,
    holistic: bool = False,
    fixed_prefix: int = 0,
    fixed_doc_ids: Sequence[str] = (),
) -> dict:
    metrics: dict[str, dict] = {}
    if holistic:
        dists = bootstrap_holistic(
            config, samples=samples, seed=seed, fixed_prefix=fixed_prefix
        )
        # holistic point estimates: one clean run over the full stream
        from lattice.config.factory import instantiate
        from lattice.harness.runner import run_on_documents
        from lattice.ports import Dataset

        documents = list(instantiate(Dataset, config.dataset).documents())
        estimates = run_on_documents(config, documents)
        for flat_key, resamples in dists.items():
            metric, key = flat_key.split(".", 1)
            # holistic BCa acceleration would need pipeline jackknife; use percentile
            pct = percentile_interval(estimates[flat_key], resamples, level)
            metrics.setdefault(metric, {})[key] = {
                "estimate": estimates[flat_key],
                "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
            }
    else:
        report_full, bundles = run_experiment_detailed(config)
        for name, bundle in bundles.items():
            dists = bootstrap(bundle, samples=samples, seed=seed, fixed_doc_ids=fixed_doc_ids)
            jacks = jackknife(bundle, fixed_doc_ids=fixed_doc_ids)
            for key, resamples in dists.items():
                estimate = report_full.metrics[name][key]
                metrics.setdefault(name, {})[key] = _iv(
                    estimate, resamples, jacks[key], level
                )
    return {
        "seed": seed,
        "level": level,
        "samples": samples,
        "config": config.model_dump(),
        "metrics": metrics,
    }


def write_report(report: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "interval-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return path
