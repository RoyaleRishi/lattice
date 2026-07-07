import dataclasses
import json
import sys

from lattice.harness.runner import run_from_path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m lattice.harness <config.toml>")
    report = run_from_path(sys.argv[1])
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
