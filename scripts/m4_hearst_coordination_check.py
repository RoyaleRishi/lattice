"""Diagnostic for Task 13 (M4 Hearst backward-coordination fix, T2),
supporting `docs/results/2026-07-31-post-fix.md` §10.2: does the fix
actually fire against the six TExEval-2 golds' corpus text, and if it fires
too rarely to move any edge count, what does "rarely" mean concretely?

Two independent checks, both against the real dataset/pipeline (not a
reimplementation of either):

1. **Connector count.** How many times `and other` / `or other` occur (as a
   whitespace-tolerant, case-insensitive regex) in each gold's raw
   `documents.jsonl` "text" fields — the naive count a reader might expect
   the fix's opportunity space to equal.
2. **Instrumented pipeline run.** Build the same `hearst`-only
   `ExperimentConfig` each gold's `m4-<gold>-sweep.toml` derives (base
   config with `relation_inducer` forced to `hearst` alone), run it through
   `run_experiment`'s own `build_orchestrator` / dataset / `process_stream`,
   and:
   - count real invocations of `_walk_coordination_backward` — i.e., how
     many `and-other`/`or-other` matches actually pair two *adjacent
     gazetteer-recognized anchors* with a fullmatching connector, which is a
     strict subset of check #1's naive substring count, since most
     occurrences sit between un-anchored running-text words;
   - count how many extra candidate pairs each invocation proposes, before
     dedup;
   - diff the induced edge set against the same run with `and-other`/
     `or-other` monkeypatched back to `coordination=False` — the exact
     pre-fix inducer, since that flag gates the only call site that invokes
     the backward walk — to get the net added/removed edge count directly,
     not inferred from an F1 delta.

    PYTHONPATH=src .venv/bin/python scripts/m4_hearst_coordination_check.py [out_dir]

Writes JSON to <out_dir>/m4-hearst-coordination-check.json (default
reports/intervals/analysis, alongside scripts/interval_analysis.py's output)
and prints a human-readable summary table.
"""

import json
import re
import sys
from pathlib import Path

import lattice.adapters.relation_inducer.hearst as hearst_module
from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig
from lattice.ports import Dataset

GOLDS = [
    "food", "food-wordnet", "science", "science-wordnet",
    "science-eurovoc", "env-eurovoc",
]
_AND_OTHER = re.compile(r"and\s+other", re.IGNORECASE)
_OR_OTHER = re.compile(r"or\s+other", re.IGNORECASE)


def _count_connector_occurrences(gold: str) -> dict[str, int]:
    path = Path("data/texeval") / gold / "documents.jsonl"
    and_other = 0
    or_other = 0
    for line in path.read_text().splitlines():
        if not line:
            continue
        text = json.loads(line)["text"]
        and_other += len(_AND_OTHER.findall(text))
        or_other += len(_OR_OTHER.findall(text))
    return {"and_other_occurrences": and_other, "or_other_occurrences": or_other}


def _hearst_only_config(gold: str) -> ExperimentConfig:
    sweep = load_config(f"configs/m4-{gold}-sweep.toml", model=SweepConfig)
    base = sweep.base.model_dump()
    base["relation_inducer"] = {"name": "hearst"}
    return ExperimentConfig.model_validate(base)


def _run_edges(config: ExperimentConfig) -> set[tuple[str, str]]:
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    orchestrator.process_stream(dataset.documents())
    snapshot = orchestrator.snapshot()
    return {(r.source_id, r.target_id) for r in snapshot.relations}


def _instrumented_check(gold: str) -> dict[str, object]:
    config = _hearst_only_config(gold)

    calls = {"n_calls": 0, "n_extra_pairs": 0}
    original_backward = hearst_module._walk_coordination_backward

    def counting_backward(anchors, last_hypo, hyper, text):
        result = original_backward(anchors, last_hypo, hyper, text)
        calls["n_calls"] += 1
        calls["n_extra_pairs"] += len(result)
        return result

    hearst_module._walk_coordination_backward = counting_backward
    try:
        after_edges = _run_edges(config)
    finally:
        hearst_module._walk_coordination_backward = original_backward

    # Emulate the pre-fix inducer: and-other/or-other shipped with
    # coordination=False, and _walk_coordination_backward did not exist.
    # Reverting the flag alone reproduces the old edge set exactly, since
    # coordination=False skips the only call site that invokes it.
    original_builtin = hearst_module._BUILTIN
    reverted = []
    for pattern in original_builtin:
        if pattern.name in ("and-other", "or-other"):
            reverted.append(
                hearst_module._Pattern(
                    pattern.name,
                    pattern.connector.pattern,
                    hyper=pattern.hyper,
                    coordination=False,
                )
            )
        else:
            reverted.append(pattern)
    hearst_module._BUILTIN = reverted
    try:
        before_edges = _run_edges(config)
    finally:
        hearst_module._BUILTIN = original_builtin

    added = sorted(after_edges - before_edges)
    removed = sorted(before_edges - after_edges)
    return {
        "backward_walk_call_sites": calls["n_calls"],
        "extra_candidate_pairs_proposed": calls["n_extra_pairs"],
        "edges_added_by_fix": len(added),
        "edges_removed_by_fix": len(removed),
        "sample_added": added[:5],
        "sample_removed": removed[:5],
    }


def run_all() -> dict[str, dict]:
    return {
        gold: {**_count_connector_occurrences(gold), **_instrumented_check(gold)}
        for gold in GOLDS
    }


def _print_summary(results: dict[str, dict]) -> None:
    header = (
        f"{'gold':<18}{'and other':>11}{'or other':>10}"
        f"{'backward calls':>16}{'extra pairs':>13}{'edges +/-':>12}"
    )
    print(header)
    for gold, row in results.items():
        edges_delta = f"+{row['edges_added_by_fix']}/-{row['edges_removed_by_fix']}"
        print(
            f"{gold:<18}{row['and_other_occurrences']:>11}"
            f"{row['or_other_occurrences']:>10}"
            f"{row['backward_walk_call_sites']:>16}"
            f"{row['extra_candidate_pairs_proposed']:>13}"
            f"{edges_delta:>12}"
        )


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/intervals/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_all()
    out_path = out_dir / "m4-hearst-coordination-check.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    _print_summary(results)
    print(f"\nwrote JSON to {out_path}")


if __name__ == "__main__":
    main()
