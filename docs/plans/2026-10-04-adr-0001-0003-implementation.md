# Implementation plan: ADR-0001, ADR-0002 and ADR-0003 follow-up code

## Context
ADRs 0001–0003 are accepted, and the specs are amended with "Implementation pending" notes. The code still reflects the old design:
- The Resolver has no lifecycle.
- The orchestrator and Engine reach the store through `getattr(resolver, "concept_store")` at `orchestrator.py:59` and `engine.py:237,247`.
- Only `skip` takes a checkpoint, and it's a full copy.
- The integrator is never checkpointed.
- Saves are v1.

This plan brings the code in line. The audit plan parked "finish T2–T6 before planning code". You asked for this plan now; whether to execute it before T2 is your call.

**Decisions taken while planning (2026-10-04):**
- **State shape.** `Resolver.snapshot()` returns `ResolverState(concepts, private)`, and `restore(ResolverState)` takes it back. The v2 save stores **only `private`** as `resolver_state`. On load, the Engine passes in the graph snapshot's concepts. Migration uses `private={}`. This relies on the invariant store concepts == graph concepts, which atomicity guarantees and a test checks.
- **Undo-log lifetime.** `checkpoint()` starts a fresh log and implicitly commits the previous one. Calling `rollback(token)` with a stale token raises. No `commit()` method is added.

If you want either recorded, I can add a short amendment to ADR-0001/0003.

## Chunks

**Status:** C6a done · C1–C3 (753b2e9) · C4 + option B (0eea93a) · C5 (ce4129f) · C6 done (9130d68) · C7 done, awaiting commit. All chunks done.

**C6a — v1 fixtures (done).** Generated from pre-change code: `tests/fixtures/saves/v1_lite_after_A_B.json` (`Engine()` + TEXT_A, TEXT_B from `tests/api/test_persistence.py`) and `v1_stemmed_after_STEM_A.json` (`_STEMMED_CONFIG` + STEM_A). Walking-skeleton baseline report saved to the session scratchpad as `baseline-walking-skeleton.json`.
The chunks run in waves. Chunks within a wave are independent and can run in parallel. Each chunk is test-first: programmer → reviewer → you commit.

### Wave 1 (parallel)
**C1. Undo-log checkpoint in `InMemoryConceptStore`** (`adapters/concept_store/in_memory.py`, `tests/contracts/concept_store_contract.py`)
- `checkpoint()` opens a log. On the first touch of an id or label since the checkpoint, `upsert` records that key's prior value (or that it was absent).
- `rollback` replays the log in reverse.
- Tests:
  - upserts, including a label change, followed by rollback
  - a second checkpoint drops the first log
  - a stale token raises
  - rollback cost doesn't depend on store size (log length == keys touched)

**C2. `GraphIntegrator.checkpoint`/`rollback`** (`ports/graph_integrator.py`, `adapters/graph_integrator/in_memory.py`, `tests/contracts/graph_integrator_contract.py`)
- Two new abstract methods.
- The in-memory version uses the same undo-log pattern over `_concepts` and `_relations`.
- Contract test: apply → checkpoint → apply → rollback, then the snapshot equals the pre-checkpoint snapshot.
- Also covers the stale-token behaviour.

**C3. Resolver lifecycle** (`ports/resolver.py`, `core/types.py`, the three resolvers, `tests/contracts/resolver_contract.py`)
- Add `ResolverState(concepts: tuple[Concept, ...], private: dict)` to `core/types.py`.
- Add abstract `checkpoint`/`rollback`/`snapshot`/`restore`/`reset` to `Resolver`.
- Add a reusable `StoreBackedResolver` base that delegates the lifecycle to `self._store`. The store becomes private: the constructor param stays `concept_store` so factory injection works.
- `exact-label` and `embedding-nn` inherit it.
- `stemmed-label` extends it so that `_concept_id_by_stem` is undo-logged, appears in `private`, is restored, and is reset.
- Contract tests:
  - resolve → checkpoint → resolve → rollback, then `snapshot()` equals the pre-checkpoint state
  - snapshot → restore into a fresh resolver round-trips
  - `restore` with `private={}` works for all three resolvers
- Update the tests that read `resolver.concept_store`: `tests/orchestrator/test_orchestrator.py:148` and `tests/config/test_factory.py:36-40`.

### Wave 2
**C4. The orchestrator checkpoints under both policies** (`orchestrator/orchestrator.py`, `tests/orchestrator/test_orchestrator.py`)
- At the start of `process()`, checkpoint the resolver and the integrator.
- On an exception, roll both back. Then `fail` re-raises, and `skip` returns a delta with the error.
- Remove the `getattr`, and rewrite the docstring at L21-34.
- Tests, for both policies: after a failing document, the graph snapshot and the resolver snapshot equal their pre-document state.
- Test: after a caught `fail`, the next document processes normally.
- Test: the integrator itself raises mid-apply, and the graph is rolled back.

### Wave 3
**C5. Save format v2** (`engine.py`, `tests/api/test_persistence.py`)
- Set `FORMAT_VERSION = 2`. `save` adds `resolver_state = resolver.snapshot().private`.
- `load` restores the integrator, then calls `resolver.restore(ResolverState(graph concepts, resolver_state))`.
- `reset` calls `resolver.reset()`. Remove the remaining `getattr(..., "concept_store")` calls. (`getattr(resolver, "embedder", None)` at `engine.py:216` stays: it isn't store access, so it's out of scope.) Fix the docstring at L150-157, since the divergence caveat is now gone.
- Update `test_save_file_shape` to version 2 with the new key.
- Add an assertion that store concepts == graph concepts after ingest.
- Test: resume-equivalence after a caught `fail`.
- Test: a v1 file raises `ValueError`.

### Wave 4
**C6. `scripts/migrate_save.py`** (new script, a v1 fixture under `tests/fixtures/`, `tests/scripts/test_migrate_save.py`)
- Per ADR-0003: read v1, build the Engine from the stored config, restore the integrator, then `resolver.restore(ResolverState(concepts, {}))` and set the counter.
- Then call `Engine.save`. Never write v2 JSON by hand. Raise clearly if the resolver can't restore from `{}`.
- Test: migrate the fixture → load → resume-equivalence against a straight-through ingest.
- Generate the fixture once from the current v1 code before C5 lands, so it's real rather than hand-written.

### Wave 5
**C7. Docs and cleanup**
- Remove the "Implementation pending" blockquotes in arch §4.2 and §8, and in M6 §4.3.
- Update the `CLAUDE.md` lines on error policy, `FORMAT_VERSION = 1` and shared-dep injection.
- Check `scripts/m3_merge_ladder_check.py:177-178`, which calls `StemmedLabelResolver.__init__(embedder=None, concept_store=None)`. It must still work.
- Mark the follow-up code done in `docs/plans/2026-10-04-spec-audit.md`.

## Ordering note
Generate C6's v1 fixture **before** merging C5. Otherwise we'd have to fake a v1 file.

## Verification
- Run `uv run ruff check .` and `uv run pytest -q -m "not ml"` after every chunk.
- After C5, also run `uv run pytest tests/api/test_persistence.py -m ml` if the models are present, to cover the standard profile.
- **No published numbers change.** Run `uv run python -m lattice.harness configs/walking-skeleton.toml` before C1 and after C5, and diff the JSON. It must be identical, apart from timing fields if there are any.
- Grep for `concept_store"` and `getattr(` in `src/` after C5. There should be no duck-typed store access left.
