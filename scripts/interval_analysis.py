"""M3 paired-delta + threshold-sensitivity curve, and the order-permutation
stability sweep, for docs/results/2026-07-14-interval-analysis.md (Task 11).

Pure orchestration over the tested library primitives — `run_experiment_detailed`,
`bootstrap`, `paired_delta`, `subsample_interval`, `order_spread` — no new
statistical algorithms are implemented here.

M3 runs on BOTH ConEL-2 and ECB+ independently (task-11-brief-v2 scope
amendment / Execution Amendment #8): the nn@0.90 operating point was chosen on
ConEL-2 in M5 and is applied uniformly (not re-tuned) to ECB+, which serves as
an out-of-sample replication check of the resolver-improvement claim.

    uv run --no-sync python scripts/interval_analysis.py [out_dir]

Writes JSON to <out_dir>/{m3-paired-delta,m3-threshold-curve,permutation-spread}.json
(gitignored, regenerable) and prints a human-readable summary to stdout.

Scheme: `clustering` is a *pooled* metric — B³ and ARI score a cross-document
pool of mentions, so they are degree-2 functionals and a with-replacement draw
manufactures self-agreeing mention pairs that inflate them. Every bootstrap
call here therefore uses scheme="subsample" (m-out-of-n, without replacement),
the same rule `lattice.harness.stats.report` applies by bundle kind, and both
constructions are the matching subsampling ones: `paired_delta(..., m=, n=)`
for the deltas and `subsample_interval` for the threshold curve. BCa is not
reachable from the subsample path (it assumes full-size bootstrap draws), so
the threshold curve's `ci_method` is `"subsample"` rather than `"bca"` and
there is no jackknife pass.

The three-way M3 comparison (exact-label -> stemmed-label ->
embedding-nn@0.90) exists because embedding-nn@0.90 makes only 7/452 merges
beyond exact-label on ConEL-2 and the Snowball stemmer reproduces 6 of the 7,
so `stemmed-label` — not `exact-label` — is the honest baseline for the
identity claim. All three pairwise deltas are emitted so the claim can be
stated against either baseline.
"""

import json
import sys
from pathlib import Path

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import (
    DeltaResult,
    Interval,
    paired_delta,
    subsample_interval,
)
from lattice.harness.stats.permutation import order_spread
from lattice.harness.stats.resample import BootstrapDraws, ResampleBundle, bootstrap

ITEM_SAMPLES = 10000  # matches the CLI's item-level default (Task 10)
BOOTSTRAP_SEED = 0  # same seed on every arm's bundle -> paired draws by construction
POOLED_SCHEME = "subsample"  # `clustering` is a pooled metric; see the module docstring
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

# The three-way identity comparison (Task 7). Order is the normalization
# ladder: raw string match -> morphological match -> embedding match.
EXACT, STEMMED, NN090 = "exact-label", "stemmed-label", "embedding-nn@0.90"
M3_PAIRS = ((NN090, EXACT), (STEMMED, EXACT), (NN090, STEMMED))

M4_GOLDS = ["env-eurovoc", "food", "food-wordnet", "science", "science-eurovoc", "science-wordnet"]
M4_CONFIG_TEMPLATE = "configs/m4-{gold}-union.toml"
M5_CONFIG = "configs/m5-conel2-nn090.toml"


def _load(path: str) -> ExperimentConfig:
    return load_config(path, model=ExperimentConfig)


def _with_resolver(base: ExperimentConfig, resolver: dict) -> ExperimentConfig:
    """A flat variant of `base` with its resolver replaced — built in-memory
    (spec amendment: only exact-label and nn@0.90 need committed collapsed TOML
    files; the threshold grid and the stemmed-label arm are programmatic.
    configs/m3-*-stemmed.toml does exist, but it is the three-way *sweep*
    config, not a collapsed single-config, so it is not an ExperimentConfig)."""
    data = base.model_dump()
    data["resolver"] = resolver
    return ExperimentConfig.model_validate(data)


def _with_threshold(base: ExperimentConfig, threshold: float) -> ExperimentConfig:
    return _with_resolver(base, {"name": "embedding-nn", "params": {"threshold": threshold}})


def _clustering_bundle(config: ExperimentConfig) -> tuple[float, ResampleBundle]:
    report, bundles = run_experiment_detailed(config)
    return report.metrics["clustering"]["b3-f1"], bundles["clustering"]


def _m3_arm_configs(corpus: str) -> dict[str, ExperimentConfig]:
    exact_cfg = _load(M3_CONFIGS[corpus]["exact-label"])
    return {
        EXACT: exact_cfg,
        STEMMED: _with_resolver(exact_cfg, {"name": "stemmed-label"}),
        NN090: _load(M3_CONFIGS[corpus]["nn@0.90"]),
    }


def m3_paired_deltas(corpus: str) -> list[dict]:
    """The three pairwise b3-f1 deltas across the identity ladder, each from
    bootstrap() run with the SAME seed and the SAME scheme on every arm's
    clustering bundle -> iteration i draws identical document indices in every
    arm (paired by construction), then paired_delta() with the arms' (m, n) so
    the size-m delta distribution is rescaled by sqrt(m / n).

    The pairing is only "by construction" while the arms enumerate documents
    identically, which is what makes the draw sequences share an RNG stream;
    that precondition is checked below rather than assumed."""
    arms: dict[str, tuple[float, BootstrapDraws]] = {}
    doc_ids: list[str] | None = None
    for label, cfg in _m3_arm_configs(corpus).items():
        estimate, bundle = _clustering_bundle(cfg)
        ids = list(bundle.per_document)
        if doc_ids is None:
            doc_ids = ids
        elif ids != doc_ids:
            raise ValueError(
                f"m3 arm {label!r} on {corpus} enumerates documents differently from the "
                "first arm, so bootstrap draws would not be paired across arms"
            )
        arms[label] = (
            estimate,
            bootstrap(bundle, samples=ITEM_SAMPLES, seed=BOOTSTRAP_SEED, scheme=POOLED_SCHEME),
        )
    rows = []
    for label_a, label_b in M3_PAIRS:
        est_a, drawn_a = arms[label_a]
        est_b, drawn_b = arms[label_b]
        delta: DeltaResult = paired_delta(
            drawn_a.draws["b3-f1"], drawn_b.draws["b3-f1"], est_a, est_b,
            level=LEVEL, m=drawn_a.m, n=drawn_a.n,
        )
        rows.append({
            "corpus": corpus,
            "pair": f"{label_a} - {label_b}",
            "arm_a": label_a,
            "arm_b": label_b,
            "b3_f1_a": est_a,
            "b3_f1_b": est_b,
            "delta_estimate": delta.estimate,
            "ci_lo": delta.lo,
            "ci_hi": delta.hi,
            "prob_positive": delta.prob_positive,
            "scheme": drawn_a.scheme,
            "m": drawn_a.m,
            "n": drawn_a.n,
            "samples": ITEM_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        })
    return rows


def m3_threshold_curve(corpus: str) -> list[dict]:
    """b3-f1 estimate + m-out-of-n subsampling CI at each threshold in
    THRESHOLD_GRID, 0.90 marked as the pre-registered (not re-tuned) operating
    point. `brackets_estimate` is necessary-not-sufficient — read it the way
    lattice.harness.stats.report._brackets documents."""
    base_cfg = _load(M3_CONFIGS[corpus]["nn@0.90"])
    rows = []
    for threshold in THRESHOLD_GRID:
        cfg = base_cfg if threshold == OPERATING_THRESHOLD else _with_threshold(base_cfg, threshold)
        estimate, bundle = _clustering_bundle(cfg)
        drawn = bootstrap(
            bundle, samples=ITEM_SAMPLES, seed=BOOTSTRAP_SEED, scheme=POOLED_SCHEME
        )
        ci: Interval = subsample_interval(
            estimate, drawn.draws["b3-f1"], m=drawn.m, n=drawn.n, level=LEVEL
        )
        rows.append({
            "corpus": corpus,
            "threshold": threshold,
            "b3_f1": estimate,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
            "ci_method": ci.method,
            "scheme": drawn.scheme,
            "m": drawn.m,
            "n": drawn.n,
            "brackets_estimate": ci.lo <= estimate <= ci.hi,
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
    m3_delta = [row for corpus in M3_CONFIGS for row in m3_paired_deltas(corpus)]
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
    print("=== M3 paired deltas (b3-f1, m-out-of-n subsampling, 95% CI) ===")
    for row in results["m3_paired_delta"]:
        print(
            f"{row['corpus']} {row['pair']}: "
            f"a={row['b3_f1_a']:.4f} b={row['b3_f1_b']:.4f} "
            f"delta={row['delta_estimate']:+.4f} "
            f"95% CI=[{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}] "
            f"prob_positive={row['prob_positive']:.4f} (m={row['m']}, n={row['n']})"
        )
    print("\n=== M3 threshold-sensitivity curve (b3-f1, subsampling 95% CI) ===")
    for corpus, rows in results["m3_threshold_curve"].items():
        print(f"-- {corpus} --")
        for row in rows:
            marker = " <= operating point" if row["is_operating_point"] else ""
            flag = "" if row["brackets_estimate"] else "  [does NOT bracket its estimate]"
            print(
                f"  threshold={row['threshold']:.2f} b3-f1={row['b3_f1']:.4f} "
                f"CI=[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]{marker}{flag}"
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
