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
    Interval,
    bca_interval,
    percentile_interval,
    subsample_interval,
)
from lattice.harness.stats.resample import bootstrap, bootstrap_holistic, jackknife
from lattice.ports import Dataset


def _brackets(estimate: float, iv: Interval) -> bool:
    """Does the interval contain the point estimate it is reported against?

    NECESSARY, NOT SUFFICIENT. Read the two values asymmetrically:

    False is decisive. The emitted band is NOT a confidence interval and must
    not be quoted as one. The usual cause is a size-dependent pooled functional
    — one whose value depends on how many distinct documents are in the sample,
    such as a set-union prediction side or a fixed corpus-level denominator
    (see EdgeF1). For those, theta(m) != theta(n) deterministically, the draws
    never straddle the estimate at any m, and the band is a corpus-size
    sensitivity range.

    True is not a clean bill of health, and in particular does not certify that
    the functional is size-invariant. A pooled subsampling band is exact only
    for a size-invariant functional; every pooled metric shipped here has some
    size dependence, and a band can bracket by a margin thin enough to be an
    artifact of the arbitrary choice m = n / 2. B3 is the live example: on M3
    ConEL-2 the raw size-29 draws for b3-recall span ~[0.8940, 0.9468] against
    a full-corpus estimate of 0.9050, so theta(m) > theta(n) systematically,
    and b3-f1's band clears its estimate only barely (see ClusteringMetric).
    ARI on the same data is genuinely well-behaved: its draws straddle at
    [0.699, 0.897] around 0.804.

    Those raw draw spans are properties of the draws and do not move with the
    rescaling factor. The clearance figure that used to be quoted here did: it
    was computed under the pre-10c sqrt(m / n), so it has been dropped rather
    than recomputed by hand. reports/intervals/ is where the current emitted
    bands live and is the only place a number like it should be read from.

    So: treat False as a hard stop, and True as "not obviously invalid" —
    then check the metric's own docstring before quoting a band as a CI.
    """
    return iv.lo <= estimate <= iv.hi


def _iv(estimate: float, resamples: list[float], jack: list[float], level: float) -> dict:
    """n-out-of-n bootstrap draws (macro metrics): BCa plus the plain
    percentile interval. BCa is the authoritative one and is what
    `brackets_estimate` is computed from. Note that BCa's own guard is not a
    guarantee of bracketing: when its adjustment would not bracket it falls
    back to the plain percentile interval, and that fallback can itself exclude
    the estimate (see test_bca_falls_back_when_interval_would_not_bracket_estimate,
    where the fallback is (10.0, 10.0) against an estimate of 5.0). The flag is
    computed from the interval that is actually emitted, so it stays correct
    either way."""
    bca = bca_interval(estimate, resamples, jack, level)
    pct = percentile_interval(estimate, resamples, level)
    return {
        "estimate": estimate,
        "bca": {"lo": bca.lo, "hi": bca.hi, "method": bca.method},
        "percentile": {"lo": pct.lo, "hi": pct.hi, "method": pct.method},
        "brackets_estimate": _brackets(estimate, bca),
    }


def _subsample_iv(
    estimate: float, subsamples: list[float], *, m: int, n: int, level: float
) -> dict:
    """m-out-of-n subsample draws (pooled metrics): the sqrt(m / (n - m))-
    rescaled interval only. BCa is wrong here — it reads the draws as a
    full-size bootstrap distribution, which is exactly what produced the
    shipped M4 reports whose intervals excluded their own point estimates. The
    raw percentile interval is not emitted either, because at the harness's
    m = round(n / 2) it is not a *different* construction from the one above
    (tau == 1; see subsample_interval) and printing it as a second column
    would suggest a corroboration that is not there."""
    iv = subsample_interval(estimate, subsamples, m=m, n=n, level=level)
    return {
        "estimate": estimate,
        "subsample": {"lo": iv.lo, "hi": iv.hi, "method": iv.method},
        "brackets_estimate": _brackets(estimate, iv),
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
