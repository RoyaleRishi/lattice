---
status: accepted
date: 2026-10-04
decision-makers: Rishi Srikaanth
---

# Error policy: every document is atomic, and the policy only decides where the error goes

Depends on [ADR-0001](0001-state-ownership-in-the-pipeline.md), under which the Resolver is a
stateful port that owns its ConceptStore.

## Context and Problem Statement

Arch spec §8 (`docs/architecture/2026-07-05-lattice-architecture-design.md` L160-170) defines
`on_error: fail | skip` but does not hold together. The 2026-10-04 spec audit (theme T1) found:

- **"Always recorded" can't hold.** L168-169 says "Regardless of policy, the error is **always**
  recorded in the `GraphDelta`", but `fail` "surfaces the exception" (L163-164). A raised
  exception returns no delta.
- **"Skipped" is undefined.** L165 says "the failing document is skipped", but earlier stages may
  already have changed state. The spec never says whether a skipped document has zero effect.
- **`fail` leaves the state undefined.** `fail` takes no checkpoint, so a caller who catches the
  exception and keeps using the Engine can leave the store holding concepts the graph never saw.
  The `engine.py:150-157` docstring admits this.
- **Integrator atomicity is accidental.** The graph integrator is not checkpointed. It is safe
  only because `apply()` runs last and the in-memory version can't fail partway
  (`orchestrator.py:21-34` docstring).
- **Policy scope is ambiguous.** M3 §7 L184-188 lists stage-specific hard errors, then says
  "everything else inherits the orchestrator `on_error` policy". It is unclear whether those
  hard errors bypass the policy.
- **The default contradicts M6.** Arch §8 calls `skip` the "future production/streaming
  default". M6 §2 L29 made `fail` the default for both profiles, citing a "partial-mutation
  caveat" that the Task 3 fix has since made stale.
- **Checkpoints copy everything.** The in-memory store's `checkpoint()` copies both of its dicts
  in full (`concept_store/in_memory.py:47-48`), so a checkpoint costs time proportional to the
  store size.

What does each policy guarantee about state, what does it do with the error, and which errors
does it cover?

## Decision Drivers

- No silent divergence between the store, the resolver and the graph, whatever the caller does
  with an exception.
- One rule that's easy to state and test, instead of a guarantee that differs per policy.
- Atomicity shouldn't depend on stage order or on how a particular integrator is written.
- No large slowdown on the default and benchmark path, which runs `fail`.
- No published numbers may change.

## Considered Options

State under `fail`:
- **X**: atomic under both policies. Checkpoint every document, roll back on failure, then
  re-raise.
- **Y**: `fail` poisons the Engine. No checkpoint; every later call raises until `load()`/`reset()`.
- **Z**: status quo, documented. State is undefined after a caught error.

How the graph integrator is made atomic:
- **(i)** `GraphIntegrator` gains `checkpoint`/`rollback`, and the orchestrator rolls it back
  alongside the resolver.
- **(ii)** the port contract requires `apply()` to be all-or-nothing, checked by a contract test.

## Decision Outcome

Chosen options: **X** and **(i)**, with checkpoints implemented as **undo logs**.

1. **Atomic per document under both policies.** At the start of each document, `process()`
   checkpoints every stateful port: the `Resolver` (which covers its `ConceptStore`, per
   ADR-0001) and the `GraphIntegrator`. On any exception it rolls both back. After that, the
   policy only decides what happens to the error:
   - `fail` re-raises it.
   - `skip` returns a `GraphDelta` with the error recorded.
2. **Both stateful ports share one lifecycle.** `GraphIntegrator` gains `checkpoint`/`rollback`
   (it already has `snapshot`/`restore` from M6). Both stateful ports now implement
   `checkpoint` / `rollback` / `snapshot` / `restore`.
3. **Checkpoints are undo logs.** Each checkpoint records only what the current document changed
   and reverses it on rollback. The cost is proportional to that document's changes, not to the
   store size. This replaces the full-copy checkpoint.
4. **Scope.** The policy covers every exception raised inside `process()`, from any stage,
   including the stage-specific hard errors in M3 §7. Errors outside `process()` (`Dataset`,
   `Metric`, `DocumentMetric`) always fail.
5. **Default stays `fail`** (M6 §2). Arch §8's "future production default" wording is amended.
   Under `fail`, the error is carried by the raised exception and the run report. Under `skip`,
   it is recorded in the delta and the run report.

Y was rejected because it removes catch-and-continue under `fail`. Z was rejected because it
leaves the known silent-corruption path open. (ii) was rejected because its atomicity depends on
`apply()` staying the final stage and on implementer discipline that a generic contract test can
only partly verify.

**Two independent rollbacks here, even though ADR-0001 rejected B2.** In B2, the resolver's
cache and the store reference each other, so they have to roll back as one unit. Here, the
resolver checkpoint and the integrator checkpoint each capture state from before the document
and restore it independently. There are no references between them, so the order of the
rollbacks doesn't matter.

**Y-statement:** In the context of an order-sensitive streaming pipeline with two stateful ports,
facing an error policy whose guarantees contradicted each other and left state undefined after a
caught `fail`, we decided to make every document atomic under both policies, by checkpointing
every stateful port with undo logs and letting the policy only route the error, to achieve one
rule with no silent divergence that holds whatever the stage order or integrator internals,
accepting a checkpoint on every document and two more methods on the GraphIntegrator port.

### Consequences

- Good: one rule. A failed document leaves no trace under either policy.
- Good: the Engine stays usable after a caught `fail`, and the store, resolver and graph can't
  diverge.
- Good: atomicity no longer depends on `apply()` being the last stage or on how an integrator
  is written.
- Good: no published numbers change, because rollback only runs when a document fails.
- Good: with undo logs, the per-document cost is proportional to that document's changes, not
  O(store size).
- Bad: every document is checkpointed, including on the default and benchmark path. Undo logs
  keep this cheap, but they are new code in the in-memory store and the in-memory integrator,
  and they have to be correct.
- Bad: the `GraphIntegrator` port grows by two methods, and every future integrator must
  implement undo-capable checkpointing.
- Neutral: the M3 gold-mentions hard error is now skipped under `skip` like any other stage
  error. The clustering metric's coverage check (M3 §7) still fails the run loudly afterwards,
  so the corpus is never silently shrunk.
- Follow-up code:
  - undo-log `checkpoint`/`rollback` for the in-memory `ConceptStore` and the in-memory
    `GraphIntegrator`
  - `GraphIntegrator.checkpoint`/`rollback` on the port, plus a contract test: process a failing
    document, then assert the graph snapshot and the resolver snapshot equal their pre-document
    state
  - the orchestrator checkpoints under both policies
  - updated docstrings at `orchestrator.py:21-34` and `engine.py:150-157`
- Follow-up spec amendments, after this ADR is accepted:
  - arch §8
  - M3 §7 (scope wording)
  - M6 §2 error-policy row (drop the stale partial-mutation caveat)
- Audit findings closed: ARCH-5, ARCH-6, M3-3, M6-1, M6-2.

## What I learned

