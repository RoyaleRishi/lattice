"""CLI: python -m lattice.harness.stats <config.toml> <out_dir> [flags]."""

import argparse

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.stats.report import analyze, write_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m lattice.harness.stats")
    parser.add_argument("config")
    parser.add_argument("out_dir")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--holistic", action="store_true")
    parser.add_argument("--fixed-prefix", type=int, default=0)
    args = parser.parse_args()
    samples = (
        args.samples if args.samples is not None else (1000 if args.holistic else 10000)
    )
    config = load_config(args.config, model=ExperimentConfig)
    report = analyze(
        config,
        samples=samples,
        seed=args.seed,
        level=args.level,
        holistic=args.holistic,
        fixed_prefix=args.fixed_prefix,
    )
    path = write_report(report, args.out_dir)
    print(f"interval report: {path}")


if __name__ == "__main__":
    main()
