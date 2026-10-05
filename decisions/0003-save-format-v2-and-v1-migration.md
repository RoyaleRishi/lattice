---
status: accepted
date: 2026-10-04
decision-makers: Rishi Srikaanth
---

# Save format v2: `load` reads only v2, and a one-off script migrates v1 files

Depends on [ADR-0001](0001-state-ownership-in-the-pipeline.md), under which the Resolver is a
stateful port that owns its ConceptStore.

## Context and Problem Statement

ADR-0001 gives the `Resolver` its own state and its own `snapshot`/`restore`. That state needs a
place in the save file, so the format moves to v2.

- **v1** (M6 §4.3) stores the config, `document_counter`, concepts and relations. It stores no
  resolver state.
- **`Engine.load` accepts exactly one version.** Any other `format_version` raises `ValueError`
  (`engine.py:27`, `engine.py:195-199`).
- **v1 can be upgraded without loss for all three current resolvers.**
  - `exact-label` and `embedding-nn` keep all their state in the store, and the store is rebuilt
    from the saved concepts. v1 load already does this with `upsert`.
  - `stemmed_label`'s `_concept_id_by_stem` cache is correctness-inert and refills as documents
    arrive.
  - So a migrated file still satisfies the M6 resume-equivalence contract.
- **We don't know whether any v1 files exist downstream** (NeuroNote).

What should `Engine.load` do with a v1 file once saves are v2?

## Decision Drivers

- Keep `Engine.load` simple and single-version.
- Don't strand v1 files, if any exist.
- Each resolver owns the shape of its state. Migration must not hand-write resolver internals.
- Resume-equivalence must hold for migrated files.

## Considered Options

- **A. Clean break.** `load` reads only v2, and v1 raises the existing `ValueError`.
- **B. Read v1, write v2.** `load` accepts both formats and upgrades v1 in memory.
- **C. Separate migration script.** `load` reads only v2, and a one-off `scripts/migrate_save.py`
  converts v1 files to v2.

## Decision Outcome

Chosen option: **C**.

**How the script works.** It reads a v1 file and builds a fresh Engine from the stored config.
It restores the integrator from the saved concepts and relations, replays the saved concepts
through the resolver's `restore`, and restores the document counter. Then it calls the normal v2
`Engine.save`. It never writes v2 JSON by hand, because each resolver owns the shape of its own
state.

**When it can't migrate.** For a resolver whose state can't be rebuilt from concepts (none
today), the script raises a clear error. It never writes a v2 file with missing state.

**Ordering.** The script depends on `Resolver.restore`/`snapshot` existing, so it is the last
chunk of the ADR-0001/0002 follow-up code, after save v2.

**Lifetime.** Delete the script once no one holds v1 files.

A was rejected because it strands any v1 saves that exist, and we don't know whether NeuroNote
holds some. B was rejected because it gives `load` a permanent second code path, and every new
resolver would have to say whether it can be rebuilt from v1, or raise. B is the classic model
where readers carry the migration logic (Kleppmann, *Designing Data-Intensive Applications*,
ch. 4). A research codebase shouldn't carry that path forever.

**Y-statement:** In the context of moving `Engine.save` to v2 so it can hold resolver state,
facing possible v1 files downstream and a `load` that accepts only one version, we decided to
keep `load` v2-only and ship a one-off script that migrates v1 files through the normal Engine
restore/save path, to achieve a simple single-version reader and an explicit, separately tested
upgrade, accepting one manual step for anyone holding v1 files and a script to delete later.

### Consequences

- Good: `Engine.load` stays single-version. The existing version check is unchanged apart from
  the version number.
- Good: the migration is explicit and has its own test. Because it goes through the normal
  restore/save path, it can't drift from the real v2 format.
- Good: v1 files aren't stranded, for any resolver whose state can be rebuilt from concepts. All
  three current resolvers qualify.
- Bad: anyone holding v1 files must run the script once before `load` works.
- Bad: one more script to maintain until it's deleted. Someone has to decide when no v1 files
  are left.
- Bad: a future resolver whose state can't be rebuilt from concepts can't migrate v1 files. The
  script raises for it rather than guessing.
- Test: migrate a v1 fixture, load the v2 result, and check resume-equivalence against
  straight-through ingest.
- Follow-up spec amendments, after this ADR is accepted:
  - M6 §2 restore-mechanism row: replace "undecided" with a pointer to this ADR
  - M6 §4.3: describe v2 and the migration script

## What I learned

