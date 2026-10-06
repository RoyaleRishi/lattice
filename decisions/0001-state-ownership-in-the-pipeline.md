---
status: accepted
date: 2026-10-04
decision-makers: Rishi Srikaanth
---

# State ownership in the pipeline: stateful ports only, and the Resolver owns its ConceptStore

## Context and Problem Statement

The architecture spec (`docs/architecture/2026-07-05-lattice-architecture-design.md`) disagrees with itself
about where pipeline state lives and who owns it. The 2026-10-04 spec audit (theme T1) found:

- **Signature.** §4.1 L87 says `process(document, memory_state) → GraphDelta`. §4.2 L95-97 says
  memory is "**not** threaded as an immutable value" and lives in stateful ports. §4 L69 rejects
  a linear pipeline precisely because it "leaks stateful memory through parameters". The code
  matches §4.2 (`process(self, document)`, `orchestrator.py:55`).
- **Which ports are stateful.** §4.2 L95 names `ConceptStore` and `GraphIntegrator`. The §6
  table marks `Resolver` and `GraphIntegrator` as stateful. §6 L128-129 calls `ConceptStore` a
  cross-cutting port.
- **Reality.** Only resolvers use the `ConceptStore`. The orchestrator and Engine reach it by
  duck typing, `getattr(resolver, "concept_store", None)` (`orchestrator.py:59`,
  `engine.py:237,247`). A resolver that names the attribute differently silently loses rollback,
  save/load and reset.
- **A resolver already holds private state.** `stemmed_label` keeps `_concept_id_by_stem`
  (`stemmed_label.py:53`), which a long docstring keeps "correctness-inert" across `rollback()`
  and `load()`. M6 §2 L32 claims "the resolver keeps no private state (both resolvers lean on the
  store)". That claim is false: there are three resolvers.

Where does pipeline state live, and who owns it?

## Decision Drivers

- Save/load, rollback under `on_error="skip"`, and `reset()` are only correct if they reach
  **every** piece of state.
- State that must stay consistent should roll back as one unit, enforced by types rather than by
  docstring discipline.
- Keep the port surface small, and keep the factory as the single composition root (spec §7.3).
- Don't break resolvers that already exist (`exact-label`, `embedding-nn`, `stemmed_label`).

## Considered Options

- **A**: only `ConceptStore` and `GraphIntegrator` are stateful. Every stage, including the
  resolver, must hold no state.
- **B1**: `Resolver` is a stateful port and **owns** its `ConceptStore`. Its lifecycle methods
  cover the store it holds.
- **B2**: `Resolver` and `ConceptStore` are two independently checkpointed stateful ports,
  coordinated by the orchestrator.
- **C**: resolvers may hold only *derived* caches, rebuilt from the store through a required
  `rebuild(store)` hook.

## Decision Outcome

Chosen option: **B1**, together with fixing the signature.

1. **(D1)** The orchestrator's unit of work is `process(document) → GraphDelta`. State lives in
   stateful ports and is never passed as a parameter. Arch spec §4.1 is corrected to match §4.2.
2. **(D2)** `Resolver` is a stateful port. It gains `checkpoint` / `rollback` / `snapshot` /
   `restore`, and it **owns** its `ConceptStore`: those four methods cover the store as well as
   any resolver-private state. The orchestrator and Engine talk only to `Resolver` and
   `GraphIntegrator`. `ConceptStore` stays a pluggable port, built by the factory and injected
   into the resolver, but it is no longer cross-cutting: only the resolver that owns it may touch
   it.

Option A was rejected because it is already violated by `stemmed_label`, and because it rules
out stateful resolvers such as centroid updates or alias tracking. Option C was rejected because
"derived only" cannot be enforced by types. Option B2 was rejected because the orchestrator would
have to coordinate two rollbacks in the right order, and the coupling between resolver state and
store would live in the spec instead of the types.

**Y-statement:** In the context of a streaming pipeline whose save/load and rollback must reach
all state, facing a resolver that already holds private state alongside a store it shares only
by duck typing, we decided to make the Resolver a stateful port that owns its ConceptStore, to
achieve one atomic unit of identity state with typed lifecycle methods, accepting that no other
component can read the store directly and that the save format must change.

### Consequences

- Good: resolver state and store state roll back, save and restore together. They cannot drift
  apart.
- Good: the `getattr(resolver, "concept_store")` duck typing goes away
  (`orchestrator.py:59`, `engine.py:237,247`).
- Good: `stemmed_label`'s cache becomes covered by a contract instead of a docstring.
- Good: future stateful resolvers (centroid updates, alias index, clustering) are allowed
  outright.
- Bad: every resolver implements four lifecycle methods. `exact-label` and `embedding-nn` mostly
  delegate them to their store.
- Bad: the `Engine.save` format must move to v2 (`FORMAT_VERSION = 1` today) to carry resolver
  state. Whether v1 files still load is not decided yet.
- Bad: no second component can read the store directly. That includes a future similarity-based
  relation inducer, two resolvers sharing one store, or a metric. The way out, if one ever
  appears, is an explicit read-only query method on `Resolver`.
- Follow-up spec amendments, after this ADR is accepted:
  - arch §4.1 (signature)
  - arch §4.2 (list of stateful ports)
  - arch §6 (the table's Stateful column and the cross-cutting paragraph)
  - M6 §2 (the restore-mechanism row)
- Audit findings closed: ARCH-1, ARCH-2, ARCH-3, M6-3. ARCH-11 is partly closed: the "graph
  store" wording still needs a §4 edit.

## What I learned
- We had a cross cutting service that could be owned by just one port
