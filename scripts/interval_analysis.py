"""M3 paired-delta + threshold-sensitivity curve, and the order-permutation
stability sweep, for docs/results/2026-07-14-interval-analysis.md (Task 11).

Pure orchestration over the tested library primitives — `run_experiment_detailed`,
`bootstrap`, `jackknife`, `paired_delta`, `bca_interval`, `order_spread` — no new
statistical algorithms are implemented here.

M3 runs on BOTH ConEL-2 and ECB+ independently (task-11-brief-v2 scope
amendment / Execution Amendment #8): the nn@0.90 operating point was chosen on
ConEL-2 in M5 and is applied uniformly (not re-tuned) to ECB+, which serves as
an out-of-sample replication check of the resolver-improvement claim.

    uv run --no-sync python scripts/interval_analysis.py [out_dir]

Writes JSON to <out_dir>/{m3-paired-delta,m3-threshold-curve,permutation-spread}.json
(gitignored, regenerable) and prints a human-readable summary to stdout.
"""

import json
import sys
from pathlib import Path

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import DeltaResult, Interval, bca_interval, paired_delta
from lattice.harness.stats.permutation import order_spread
from lattice.harness.stats.resample import ResampleBundle, bootstrap, jackknife

ITEM_SAMPLES = 10000  # matches the CLI's item-level default (Task 10)
BOOTSTRAP_SEED = 0  # same seed on both configs' bundles -> paired draws by construction
PERMUTATIONS = 40
PERMUTATION_SEED = 1
LEVEL = 0.95
THRESHOLD_GRID = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
OPERATING_THRESHOLD = 0.90  # pre-registered on ConEL-2 in M5; applied uniformly, not re-tuned

M3_CONFIGS = {
    "conel2": {
        "exact-label": "configs/m3-conel2-exact.toml",
        "nn@0.90": "configs/m3-conel2-nn090.toml",
    },
    "ecbplus": {
        "exact-label": "configs/m3-ecbplus-exact.toml",
        "nn@0.90": "configs/m3-ecbplus-nn090.toml",
    },
}

M4_GOLDS = ["env-eurovoc", "food", "food-wordnet", "science", "science-eurovoc", "science-wordnet"]
M4_CONFIG_TEMPLATE = "configs/m4-{gold}-union.toml"
M5_CONFIG = "configs/m5-conel2-nn090.toml"


def _load(path: str) -> ExperimentConfig:
    return load_config(path, model=ExperimentConfig)


def _with_threshold(base: ExperimentConfig, threshold: float) -> ExperimentConfig:
    """A flat embedding-nn variant of `base` at `threshold` — built in-memory
    (spec amendment: only exact-label and nn@0.90 need committed TOML files;
    the rest of the threshold grid is programmatic)."""
    data = base.model_dump()
    data["resolver"] = {"name": "embedding-nn", "params": {"threshold": threshold}}
    return ExperimentConfig.model_validate(data)


def _clustering_bundle(config: ExperimentConfig) -> tuple[float, ResampleBundle]:
    report, bundles = run_experiment_detailed(config)
    return report.metrics["clustering"]["b3-f1"], bundles["clustering"]


def m3_paired_delta(corpus: str) -> dict:
    """nn@0.90 - exact-label on b3-f1, via bootstrap() run with the SAME seed
    on both configs' clustering bundles -> iteration i draws identical
    document indices (paired by construction), then paired_delta()."""
    exact_cfg = _load(M3_CONFIGS[corpus]["exact-label"])
    nn090_cfg = _load(M3_CONFIGS[corpus]["nn@0.90"])
    est_exact, exact_bundle = _clustering_bundle(exact_cfg)
    est_nn090, nn090_bundle = _clustering_bundle(nn090_cfg)
    exact_resamples = bootstrap(exact_bundle, samples=ITEM_SAMPLES, seed=BOOTSTRAP_SEED)
    nn090_resamples = bootstrap(nn090_bundle, samples=ITEM_SAMPLES, seed=BOOTSTRAP_SEED)
    delta: DeltaResult = paired_delta(
        nn090_resamples["b3-f1"], exact_resamples["b3-f1"], est_nn090, est_exact, level=LEVEL
    )
    return {
        "corpus": corpus,
        "exact_label_b3_f1": est_exact,
        "nn090_b3_f1": est_nn090,
        "delta_estimate": delta.estimate,
        "ci_lo": delta.lo,
        "ci_hi": delta.hi,
        "prob_positive": delta.prob_positive,
        "samples": ITEM_SAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def m3_threshold_curve(corpus: str) -> list[dict]:
    """b3-f1 estimate + BCa CI at each threshold in THRESHOLD_GRID, 0.90
    marked as the pre-registered (not re-tuned) operating point."""
    base_cfg = _load(M3_CONFIGS[corpus]["nn@0.90"])
    rows = []
    for threshold in THRESHOLD_GRID:
        cfg = base_cfg if threshold == OPERATING_THRESHOLD else _with_threshold(base_cfg, threshold)
        estimate, bundle = _clustering_bundle(cfg)
        resamples = bootstrap(bundle, samples=ITEM_SAMPLES, seed=BOOTSTRAP_SEED)
        jack = jackknife(bundle)
        ci: Interval = bca_interval(estimate, resamples["b3-f1"], jack["b3-f1"], level=LEVEL)
        rows.append({
            "corpus": corpus,
            "threshold": threshold,
            "b3_f1": estimate,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
            "ci_method": ci.method,
            "is_operating_point": threshold == OPERATING_THRESHOLD,
        })
    return rows


def _spread_row(spreads: dict, key: str) -> dict:
    s = spreads[key]
    return {"min": s.min, "max": s.max, "range": s.range, "std": s.std}


def permutation_spread(path: str, *, fixed_prefix: int, label: str) -> dict:
    cfg = _load(path)
    spreads = order_spread(cfg, permutations=PERMUTATIONS, seed=PERMUTATION_SEED,
                            fixed_prefix=fixed_prefix)
    return {
        "label": label,
        "fixed_prefix": fixed_prefix,
        "permutations": PERMUTATIONS,
        "seed": PERMUTATION_SEED,
        "keys": {key: _spread_row(spreads, key) for key in spreads},
    }


def run_all() -> dict:
    m3_delta = [m3_paired_delta(corpus) for corpus in M3_CONFIGS]
    m3_curve = {corpus: m3_threshold_curve(corpus) for corpus in M3_CONFIGS}
    permutations = []
    # M3: nn@0.90 (the operating point under test) per corpus, fixed_prefix=0
    # (M3 has no glossary-first constraint).
    for corpus, configs in M3_CONFIGS.items():
        permutations.append(
            permutation_spread(configs["nn@0.90"], fixed_prefix=0, label=f"m3-{corpus}-nn090")
        )
    # M4: six golds, fixed_prefix=1 holds the glossary document (stream
    # position 0) fixed while the rest of the corpus is shuffled.
    for gold in M4_GOLDS:
        permutations.append(
            permutation_spread(
                M4_CONFIG_TEMPLATE.format(gold=gold), fixed_prefix=1, label=f"m4-{gold}-union"
            )
        )
    # M5: fixed_prefix=0, the full-real holistic pipeline.
    permutations.append(permutation_spread(M5_CONFIG, fixed_prefix=0, label="m5-conel2-nn090"))
    return {"m3_paired_delta": m3_delta, "m3_threshold_curve": m3_curve,
            "permutation_spread": permutations}


def _print_summary(results: dict) -> None:
    print("=== M3 paired delta (nn@0.90 - exact-label, b3-f1) ===")
    for row in results["m3_paired_delta"]:
        print(
            f"{row['corpus']}: exact={row['exact_label_b3_f1']:.4f} "
            f"nn090={row['nn090_b3_f1']:.4f} delta={row['delta_estimate']:+.4f} "
            f"95% CI=[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] "
            f"prob_positive={row['prob_positive']:.4f}"
        )
    print("\n=== M3 threshold-sensitivity curve (b3-f1, BCa 95% CI) ===")
    for corpus, rows in results["m3_threshold_curve"].items():
        print(f"-- {corpus} --")
        for row in rows:
            marker = " <= operating point" if row["is_operating_point"] else ""
            print(
                f"  threshold={row['threshold']:.2f} b3-f1={row['b3_f1']:.4f} "
                f"CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]{marker}"
            )
    print(f"\n=== Order-permutation spread (K={PERMUTATIONS}, seed={PERMUTATION_SEED}) ===")
    for entry in results["permutation_spread"]:
        print(f"-- {entry['label']} (fixed_prefix={entry['fixed_prefix']}) --")
        for key, stats in entry["keys"].items():
            print(f"  {key}: range={stats['range']:.6f} std={stats['std']:.6f}")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/intervals/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_all()
    (out_dir / "m3-paired-delta.json").write_text(
        json.dumps(results["m3_paired_delta"], indent=2, sort_keys=True)
    )
    (out_dir / "m3-threshold-curve.json").write_text(
        json.dumps(results["m3_threshold_curve"], indent=2, sort_keys=True)
    )
    (out_dir / "permutation-spread.json").write_text(
        json.dumps(results["permutation_spread"], indent=2, sort_keys=True)
    )
    _print_summary(results)
    print(f"\nwrote JSON to {out_dir}/")


if __name__ == "__main__":
    main()
