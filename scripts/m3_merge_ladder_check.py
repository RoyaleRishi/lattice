"""Diagnostic for Task 14 (README identity claim), supporting
`docs/results/2026-07-31-post-fix.md` §3.1 and the README's opening pitch:
*how many merges does the shipped `embedding-nn@0.90` resolver actually make
beyond exact string matching on ConEL-2, and how many of those does a free
Snowball stemmer reproduce?*

§3.1 states "7 of 452" and "6 of those 7" in prose, and
`StemmedLabelResolver`'s docstring repeats them, but no artifact under
`reports/` carried them — they were the last headline figures in the README
with no regenerable file behind them. This script closes that gap the way
Task 13 closed its own (`scripts/m4_hearst_coordination_check.py`): by
running the real pipeline rather than reimplementing it, and archiving the
output next to `scripts/interval_analysis.py`'s.

Method. All three arms of `configs/m3-conel2-stemmed.toml` are run through
the harness's own `build_orchestrator` / dataset / `process_stream` — the
same code path the sweep uses — and the per-mention concept assignment is
read off `GraphDelta.resolutions` in stream order. Then:

1. **Alignment is asserted, not assumed** (Task 10b's precedent). The three
   arms must enumerate the identical mention sequence, keyed on
   `(unit_id, span, surface)`; the extractor is `gold-mentions` and the
   scorer `passthrough`, so they should, but a silent divergence would make
   every count below meaningless.
2. **Merge counts.** `merges = mentions - concepts` per arm, and the
   `embedding-nn` advantage as `concepts(exact) - concepts(nn)`.
3. **Which merges.** Every `embedding-nn` concept is mapped to the set of
   `exact-label` concepts whose mentions it absorbed. A group spanning *k*
   exact-label concepts contributes *k - 1* extra merges. (The script also
   asserts no exact-label group is *split* by `embedding-nn`, which cannot
   happen — identical labels embed identically, so cosine 1.0 >= 0.90 — but
   an unasserted "cannot happen" is how §3.1's figure became unverifiable in
   the first place.)
4. **Stem reproduction.** Within each such group the same surfaces are
   re-partitioned by Snowball stem, using `StemmedLabelResolver`'s own
   `_stem`. Merges the stemmer reproduces are counted, and the ones it misses
   are listed by surface so the reader can see what the embedding bought.

    PYTHONPATH=src .venv/bin/python scripts/m3_merge_ladder_check.py [out_dir]

Writes JSON to <out_dir>/m3-merge-ladder-check.json (default
reports/intervals/analysis) and prints a human-readable summary.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from lattice.adapters.resolver.stemmed_label import StemmedLabelResolver
from lattice.config.factory import build_orchestrator, instantiate
from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig
from lattice.ports import Dataset

CONFIG = "configs/m3-conel2-stemmed.toml"
ARMS = {
    "exact-label": {"name": "exact-label"},
    "stemmed-label": {"name": "stemmed-label"},
    "embedding-nn@0.90": {"name": "embedding-nn", "params": {"threshold": 0.90}},
}

MentionKey = tuple[str, int, int, str]


def _arm_config(resolver: dict) -> ExperimentConfig:
    sweep = load_config(CONFIG, model=SweepConfig)
    base = sweep.base.model_dump()
    base["resolver"] = resolver
    return ExperimentConfig.model_validate(base)


def _run_arm(resolver: dict) -> tuple[list[MentionKey], list[str], int]:
    """Returns (mention keys in stream order, concept id per mention,
    concept count from the graph snapshot)."""
    config = _arm_config(resolver)
    orchestrator = build_orchestrator(config)
    dataset = instantiate(Dataset, config.dataset)
    deltas = orchestrator.process_stream(dataset.documents())
    keys: list[MentionKey] = []
    concept_ids: list[str] = []
    for delta in deltas:
        for resolution in delta.resolutions:
            mention = resolution.mention.mention
            keys.append(
                (mention.unit_id, mention.span[0], mention.span[1], mention.surface)
            )
            concept_ids.append(resolution.concept.id)
    return keys, concept_ids, len(orchestrator.snapshot().concepts)


def run_all() -> dict[str, object]:
    arms = {name: _run_arm(resolver) for name, resolver in ARMS.items()}

    # (1) alignment
    reference_keys = arms["exact-label"][0]
    for name, (keys, _, _) in arms.items():
        if keys != reference_keys:
            raise AssertionError(
                f"arm {name!r} does not enumerate the same mentions as "
                f"'exact-label' ({len(keys)} vs {len(reference_keys)}); "
                "every count in this report assumes a shared mention sequence"
            )
    n_mentions = len(reference_keys)

    # (2) merge counts
    per_arm = {}
    for name, (_, concept_ids, snapshot_count) in arms.items():
        distinct = len(set(concept_ids))
        if distinct != snapshot_count:
            raise AssertionError(
                f"arm {name!r}: {distinct} distinct resolution concept ids but "
                f"{snapshot_count} concepts in the snapshot"
            )
        per_arm[name] = {
            "mentions": n_mentions,
            "concepts": snapshot_count,
            "merges": n_mentions - snapshot_count,
        }

    exact_ids = arms["exact-label"][1]
    nn_ids = arms["embedding-nn@0.90"][1]
    extra_merges = per_arm["exact-label"]["concepts"] - per_arm["embedding-nn@0.90"]["concepts"]

    # (3) which exact-label concepts each embedding-nn concept absorbed
    exact_label_of: dict[str, str] = {}
    for key, exact_id in zip(reference_keys, exact_ids, strict=True):
        exact_label_of.setdefault(exact_id, key[3].strip().lower())
    nn_to_exact: dict[str, set[str]] = defaultdict(set)
    exact_to_nn: dict[str, set[str]] = defaultdict(set)
    for exact_id, nn_id in zip(exact_ids, nn_ids, strict=True):
        nn_to_exact[nn_id].add(exact_id)
        exact_to_nn[exact_id].add(nn_id)
    split = {e: sorted(n) for e, n in exact_to_nn.items() if len(n) > 1}
    if split:
        raise AssertionError(
            f"embedding-nn split {len(split)} exact-label group(s); identical "
            "labels must embed identically, so this should be impossible"
        )

    # (4) stem reproduction inside each merged group
    stemmer = StemmedLabelResolver.__new__(StemmedLabelResolver)
    StemmedLabelResolver.__init__(stemmer, embedder=None, concept_store=None)
    groups = []
    reproduced = 0
    for nn_id, members in nn_to_exact.items():
        if len(members) < 2:
            continue
        surfaces = sorted(exact_label_of[e] for e in members)
        by_stem: dict[str, list[str]] = defaultdict(list)
        for surface in surfaces:
            by_stem[stemmer._stem(surface)].append(surface)
        merges_here = len(surfaces) - 1
        stem_merges = len(surfaces) - len(by_stem)
        reproduced += stem_merges
        groups.append(
            {
                "surfaces": surfaces,
                "merges": merges_here,
                "reproduced_by_stemmer": stem_merges,
                "stem_partition": {k: v for k, v in sorted(by_stem.items())},
                "fully_reproduced": stem_merges == merges_here,
            }
        )
    groups.sort(key=lambda g: g["surfaces"])
    counted = sum(g["merges"] for g in groups)
    if counted != extra_merges:
        raise AssertionError(
            f"enumerated {counted} extra merges but the concept-count "
            f"difference is {extra_merges}"
        )

    return {
        "corpus": "conel2/test",
        "config": CONFIG,
        "mentions": n_mentions,
        "arms": per_arm,
        "embedding_nn_merges_beyond_exact_label": extra_merges,
        "of_which_reproduced_by_snowball_stem": reproduced,
        "merged_groups": groups,
        "not_reproduced_by_stemmer": [
            g["surfaces"] for g in groups if not g["fully_reproduced"]
        ],
    }


def _print_summary(results: dict) -> None:
    print(f"{results['corpus']}: {results['mentions']} gold mentions\n")
    print(f"{'arm':<20}{'concepts':>10}{'merges':>9}")
    for name, row in results["arms"].items():
        print(f"{name:<20}{row['concepts']:>10}{row['merges']:>9}")
    extra = results["embedding_nn_merges_beyond_exact_label"]
    same = results["of_which_reproduced_by_snowball_stem"]
    print(
        f"\nembedding-nn@0.90 makes {extra} merges beyond exact-label "
        f"({extra}/{results['mentions']} mentions);"
        f"\nthe Snowball stemmer reproduces {same} of those {extra}."
    )
    for surfaces in results["not_reproduced_by_stemmer"]:
        print(f"  not reproduced: {' | '.join(surfaces)}")


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/intervals/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_all()
    out_path = out_dir / "m3-merge-ladder-check.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    _print_summary(results)
    print(f"\nwrote JSON to {out_path}")


if __name__ == "__main__":
    main()
