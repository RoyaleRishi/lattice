# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

lattice is a concept-memory engine: a stream of documents in, an accreting, normalized concept graph out. Concept identity is carried **across** documents (not re-extracted per document). It is also a research codebase: every published number in `README.md` is backed by an artifact under `reports/`, and the README is deliberately honest about what is and isn't measured.

## Commands

Python ≥ 3.12, managed with `uv`.

```bash
uv sync                                   # core + dev (what CI installs)
uv sync --group ml                        # + spaCy, sentence-transformers, datasets
uv run python scripts/fetch_models.py     # one-time model download for `ml` tests / "standard" profile

uv run ruff check .                       # lint (E, F, I, UP; line length 100)
uv run pytest -q -m "not ml"              # what CI runs (3.12 and 3.13)
uv run pytest tests/adapters/test_exact_label_resolver.py            # one file
uv run pytest tests/adapters/test_exact_label_resolver.py::TestExactLabelResolver::test_labels_are_normalized_lowercase
```

Tests marked `ml` (module-level `pytestmark = pytest.mark.ml` or per-test) need the `ml` group, downloaded models, and/or fetched datasets under `data/` (gitignored). CI excludes them on purpose.

Experiment harness:

```bash
uv run python -m lattice.harness configs/walking-skeleton.toml          # single run -> JSON report on stdout
uv run python -m lattice.harness --sweep configs/m3-conel2-sweep.toml [out_dir]   # writes sweep-report.{json,md}, default out_dir=reports
uv run python -m lattice.harness.stats <config.toml> <out_dir> [--holistic] [--samples N] [--seed S] [--fixed-prefix K]
```

Datasets are fetched and converted once to plain JSONL by `scripts/fetch_*.py` (e.g. `uv run --group ml python scripts/fetch_datasets.py inspec`). Adapters only ever read that JSONL. Analysis scripts in `scripts/` (`interval_analysis.py`, `m3_merge_ladder_check.py`, `m4_hearst_coordination_check.py`, `promptrank_analysis.py`) regenerate files under `reports/intervals/analysis/`.

## Architecture (ports & adapters)

**Pipeline**: six stages per document, run by `Orchestrator.process()` (`src/lattice/orchestrator/orchestrator.py`):
segmenter → extractor → scorer → (keep `selected` mentions) → resolver → relation_inducer → graph_integrator. The output is a `GraphDelta`. Batch processing is just a fold over `process()`; there is no separate batch path. Ingestion is order-sensitive because the resolver merges greedily in arrival order.

- `src/lattice/ports/`: one ABC per stage, plus support ports: `Embedder`, `ConceptStore` (owned by the resolver, ADR-0001), `Dataset`, `Metric` (scores the final `GraphSnapshot`) and `DocumentMetric` (scores the per-document `GraphDelta`s).
- `src/lattice/adapters/<port>/`: concrete implementations. Each one self-registers with `@register(Port, "name")` (`src/lattice/registry/registry.py`). **To add an adapter, write the decorated class and import its module in `src/lattice/adapters/__init__.py`.** Nothing else changes.
- `src/lattice/config/factory.py` is the **single composition root**. It maps validated TOML/dict config (`config/schema.py`) through registry lookup to instances. Shared deps are injected automatically unless config sets them explicitly: `embedder` into any adapter whose `__init__` has a parameter with that name; `concept_store` the same way. By ADR-0001 only resolvers declare that parameter, because the resolver owns the store. This is a convention, not something the factory enforces.
- `src/lattice/engine.py`: public facade `Engine`, with `lite` and `standard` profiles that share one topology. `lite` (token extractor + hashing embedder) is dependency-free but its 0.90 resolver threshold is **uncalibrated**. Treat it as a smoke-test profile only. `Engine.from_config(...)` uses the same schema as the harness. `save()`/`load()` write versioned JSON (`FORMAT_VERSION = 2`: config, graph, document counter, and resolver-private state as `resolver_state`) and must resume exactly (test-enforced). `load` reads v2 only; convert v1 files once with `uv run python -m scripts.migrate_save <v1.json> <v2.json>` (ADR-0003).
- Public API is exactly `lattice.__all__`. Everything else is internal.
- `src/lattice/harness/`: `runner.py` (`ExperimentConfig` = `RunConfig` + `dataset`, `metrics`, `document_metrics`), `sweep.py` (`[base.*]` config + `[axes]` that vary only `scorer` / `resolver` / `relation_inducer`), `stats/` (bootstrap/subsample intervals, paired deltas, permutation order-sensitivity; `--holistic` re-runs the whole pipeline per resample for snapshot-level metrics).
- Error policy `run.on_error`: every document is atomic under both policies (ADR-0002): `process()` checkpoints the resolver (which owns its `ConceptStore`) and the graph integrator, and rolls both back on any failure, including `KeyboardInterrupt`. Then `"fail"` re-raises and `"skip"` records the error in the delta.

## Testing conventions

- `tests/contracts/<port>_contract.py` defines a contract class per port. Each adapter test subclasses it (`class TestX(ResolverContract)`) and implements the factory method (e.g. `make_resolver()`), then adds adapter-specific tests. New adapters should do the same.
- Build test objects with `tests/helpers.py` (`make_document`, `make_scored_mention`, …). Use small fixtures in `tests/fixtures/`.
- Use the `clean_registry` fixture (`tests/conftest.py`) for any test that registers throwaway adapters.
- `tests/api/test_readme.py` executes the **first** ```python block in `README.md` verbatim. Editing the README quickstart can break tests.

## Results discipline

- `reports/**` is gitignored except `*.json` / `*.md`, because those back the README numbers. If you change an adapter or metric that a published number depends on, re-run the matching config/sweep and update the report and the README together. Don't hand-edit numbers.
- Report the pre-registered operating point (e.g. resolver `embedding-nn @ 0.90`), not the best point on a threshold grid. See `docs/2026-07-14-statistical-intervals-design.md` §6.
- Interval cells with `brackets_estimate: false` are size-sensitivity ranges, not confidence intervals.
- When a config uses the `redundancy` metric, its `threshold` must be below the resolver's threshold, or it measures nothing (see the circularity note in `adapters/metric/redundancy.py` and the comments in `configs/m5-*.toml`).
- The README's "What is not measured, and what is stale" section lists known-stale artifacts (PromptRank reports, three M5 holistic interval reports). Don't quote them.

## Docs layout

Design specs are `docs/YYYY-MM-DD-<milestone>-design.md` (start with `docs/2026-07-05-lattice-architecture-design.md`). Implementation plans are in `docs/plans/`. Result write-ups, including negative results, are in `docs/results/`. Code comments cite spec sections (e.g. "spec §7.3"); keep that convention. Milestones: M2 extraction/salience, M3 normalization (identity), M4 hierarchy, M5 integration, M6 API hardening.
