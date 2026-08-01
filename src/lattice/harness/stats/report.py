"""Assemble the interval report from bundles + engine + intervals, and write
it as regenerable JSON. Item-level metrics bootstrap from one pipeline run;
holistic metrics re-run the pipeline per resample."""

import json
from collections.abc import Sequence
from pathlib import Path

from lattice.config.factory import instantiate
from lattice.harness.runner import (
    ExperimentConfig,
    run_experiment_detailed,
    run_on_documents,
)
from lattice.harness.stats.intervals import (
    bca_interval,
    percentile_interval,
    subsample_interval,
)
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife
from lattice.ports import Dataset


def _iv(estimate: float, resamples: list[float], jack: list[float], level: float) -> dict:
    """n-out-of-n bootstrap draws (macro metrics): BCa plus the plain
    percentile interval."""
    bca = bca_interval(estimate, resamples, jack, level)
    pct = percentile_interval(estimate, resamples, level)
    return {
        "estimate": estimate,
        "bca": {"lo": bca.lo, "hi": bca.hi, "method": bca.method},
        "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
    }


def _subsample_iv(
    estimate: float, subsamples: list[float], *, m: int, n: int, level: float
) -> dict:
    """m-out-of-n subsample draws (pooled metrics): the sqrt(m / n)-rescaled
    interval only. BCa and the raw percentile interval are both wrong here —
    they read the draws as a full-size bootstrap distribution, which is exactly
    what produced the shipped M4 reports whose intervals excluded their own
    point estimates."""
    iv = subsample_interval(estimate, subsamples, m=m, n=n, level=level)
    return {
        "estimate": estimate,
        "subsample": {"lo": iv.lo, "hi": iv.hi, "method": iv.method},
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
            fixed_ids = list(fixed_doc_ids) or list(bundle.per_document)[:fixed_prefix]
            # Scheme by metric kind. Macro metrics average a per-document score
            # and are a correct classical bootstrap. Pooled metrics score a
            # cross-document pool of mentions/edges, where a duplicate document
            # is not a second observation but a self-agreeing artifact, so they
            # take m-out-of-n subsampling instead (see resample.py).
            scheme = "subsample" if bundle.kind == "pooled" else "resample"
            drawn = bootstrap(
                bundle, samples=samples, seed=seed, fixed_doc_ids=fixed_ids, scheme=scheme
            )
            # jackknife feeds BCa's acceleration, which only applies to the
            # n-out-of-n path.
            jacks = jackknife(bundle, fixed_doc_ids=fixed_ids) if scheme == "resample" else {}
            for key, draws in drawn.draws.items():
                estimate = report_full.metrics[name][key]
                entry = (
                    _subsample_iv(estimate, draws, m=drawn.m, n=drawn.n, level=level)
                    if scheme == "subsample"
                    else _iv(estimate, draws, jacks[key], level)
                )
                entry |= {"scheme": drawn.scheme, "m": drawn.m, "n": drawn.n}
                metrics.setdefault(name, {})[key] = entry
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
