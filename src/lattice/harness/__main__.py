import dataclasses
import json
import sys

from lattice.config.loader import load_config
from lattice.harness.runner import run_from_path
from lattice.harness.sweep import SweepConfig, run_sweep, write_reports


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--sweep":
        if len(args) not in (2, 3):
            raise SystemExit("usage: python -m lattice.harness --sweep <sweep.toml> [out_dir]")
        sweep = load_config(args[1], model=SweepConfig)
        report = run_sweep(sweep)
        json_path, md_path = write_reports(report, args[2] if len(args) == 3 else "reports")
        print(md_path.read_text())
        print(f"reports: {json_path} {md_path}")
        return
    if len(args) != 1:
        raise SystemExit(
            "usage: python -m lattice.harness <config.toml> | --sweep <sweep.toml> [out_dir]"
        )
    print(json.dumps(dataclasses.asdict(run_from_path(args[0])), indent=2))


if __name__ == "__main__":
    main()
