"""One-off migration of lattice save files from format v1 to v2 (ADR-0003).

    .venv/bin/python -m scripts.migrate_save <v1.json> <v2.json>

Builds a fresh Engine from the stored config, restores the graph and the
resolver from the saved concepts (v1 has no resolver state), then writes the
file with the normal Engine.save. No v2 JSON is written by hand, so the output
can't drift from the real format. If a resolver can't rebuild from concepts
alone, this raises and writes nothing.

Delete this script once no v1 files remain.
"""

import argparse
import json
from pathlib import Path

from lattice.config.schema import RunConfig
from lattice.core.types import GraphSnapshot, ResolverState
from lattice.engine import Engine


def migrate(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    if dst.resolve() == src.resolve():
        raise ValueError("refusing to migrate in place; give a different output path")
    payload = json.loads(src.read_text())
    found = payload.get("format_version")
    if found == 2:
        raise ValueError(f"{src} is already format_version 2; nothing to migrate")
    if found != 1:
        raise ValueError(f"expected format_version 1, got {found!r}")

    engine = Engine.__new__(Engine)
    engine._init(RunConfig.model_validate(payload["config"]), payload["profile"])
    concepts, relations = engine._graph_from_payload(payload)
    orchestrator = engine._orchestrator
    orchestrator.graph_integrator.restore(GraphSnapshot(concepts=concepts, relations=relations))
    try:
        orchestrator.resolver.restore(ResolverState(concepts, {}))
    except Exception as exc:
        name = type(orchestrator.resolver).__name__
        raise RuntimeError(
            f"{name} cannot rebuild its state from concepts alone; v1 file not migrated"
        ) from exc
    engine._counter = int(payload["document_counter"])
    engine.save(dst)  # only after everything restored, so a failure leaves no file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path, help="v1 save file")
    parser.add_argument("dst", type=Path, help="where to write the v2 file")
    args = parser.parse_args()
    migrate(args.src, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
