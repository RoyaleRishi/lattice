"""PromptRank-vs-incumbent paired-delta bootstrap on Inspec f1@k, for
docs/results/2026-07-29-promptrank-baseline.md (credibility Track 3).

Pure orchestration over run_experiment_detailed + bootstrap + paired_delta —
the same primitives scripts/interval_analysis.py uses for M3's paired delta;
no new statistical algorithms. Both configs run on the same 500-document
Inspec test split, so the same seed draws identical document indices at each
iteration (paired by construction).

    uv run --no-sync python scripts/promptrank_analysis.py \
        [out_dir] --incumbent configs/m2b-f1atk.toml --incumbent-label embedding-cosine
"""

import argparse
import json
from pathlib import Path

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import DeltaResult, paired_delta
from lattice.harness.stats.resample import bootstrap

PROMPTRANK_CONFIG = "configs/m2-promptrank-f1atk.toml"
ITEM_SAMPLES = 10000  # matches the CLI's item-level default (Track 1)
SEED = 0
LEVEL = 0.95
KS = ["f1@5", "f1@10", "f1@15"]


def _f1atk(path: str):
    cfg = load_config(path, model=ExperimentConfig)
    report, bundles = run_experiment_detailed(cfg)
    return report.metrics["f1-at-k"], bundles["f1-at-k"]


def paired(incumbent_path: str, incumbent_label: str) -> dict:
    pr_est, pr_bundle = _f1atk(PROMPTRANK_CONFIG)
    in_est, in_bundle = _f1atk(incumbent_path)
    # f1-at-k is a macro metric (per-document mean), so the classical
    # n-out-of-n bootstrap is the right scheme and these numbers are unchanged
    # by the pooled-metric subsampling fix.
    pr_res = bootstrap(pr_bundle, samples=ITEM_SAMPLES, seed=SEED).draws
    in_res = bootstrap(in_bundle, samples=ITEM_SAMPLES, seed=SEED).draws
    rows = []
    for k in KS:
        d: DeltaResult = paired_delta(pr_res[k], in_res[k], pr_est[k], in_est[k], level=LEVEL)
        rows.append({
            "metric": k,
            "promptrank": pr_est[k],
            "incumbent": in_est[k],
            "delta_estimate": d.estimate,
            "ci_lo": d.lo,
            "ci_hi": d.hi,
            "prob_positive": d.prob_positive,
        })
    return {"incumbent_label": incumbent_label, "samples": ITEM_SAMPLES, "seed": SEED, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(prog="promptrank_analysis")
    p.add_argument("out_dir", nargs="?", default="reports/intervals/promptrank")
    p.add_argument("--incumbent", default="configs/m2b-f1atk.toml")
    p.add_argument("--incumbent-label", default="embedding-cosine")
    args = p.parse_args()
    result = paired(args.incumbent, args.incumbent_label)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promptrank-paired-delta.json").write_text(json.dumps(result, indent=2))
    print(f"=== PromptRank - {args.incumbent_label} paired delta (Inspec f1@k) ===")
    for r in result["rows"]:
        print(
            f"{r['metric']}: promptrank={r['promptrank']:.4f} "
            f"{args.incumbent_label}={r['incumbent']:.4f} "
            f"delta={r['delta_estimate']:+.4f} "
            f"95% CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
            f"prob_positive={r['prob_positive']:.4f}"
        )


if __name__ == "__main__":
    main()
