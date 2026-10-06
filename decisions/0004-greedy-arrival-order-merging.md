---
status: proposed
date: 2026-10-05
decision-makers: Rishi Srikaanth
---

# Greedy, arrival-order resolver merging; batch is a fold over `process()`

Retroactive. Recorded 2026-10-05 for decisions made in the architecture spec (2026-07-05),
the orchestrator (`66f0705`, 2026-07-07) and the embedding-NN resolver (`7ca722a`, 2026-07-11).
Related: [ADR-0001](0001-state-ownership-in-the-pipeline.md) (the Resolver owns its ConceptStore).

## Context and Problem Statement

lattice ingests a stream of documents and carries concept identity across them. Something has
to decide, for each new mention, whether it is an existing concept or a new one, and whether
batch ingestion gets its own code path.

- **Spec.** "Batch is a fold over the stream … Batch has no privileged code path; this
  invariant is what keeps the streaming case first-class rather than bolted on"
  (arch spec :91-92). Ports are "streaming-native from day one" (:93-94).
- **M3.** Mentions resolve "in input order" (M3 :47). A mention merges when the top hit's
  cosine ≥ `threshold` (:52-53). Concept embeddings are fixed at creation; centroid updates
  are a documented deferral (:61-64).
- **Code.** `process_stream` is a list fold over `process()` (`orchestrator.py:121-122`).
  `embedding_nn.py:35-51` tries an exact-label hit, then nearest k=1, then mints a uuid5
  concept. Store ties break by concept id (`in_memory.py:59`).
- **Consequence: output depends on arrival order.** K = 40 permutations, seed 1
  (README :272-301): ConEL-2 b3-f1 range 3.3e-16 (float residue); ECB+ b3-f1 0.0037,
  b3-recall 0.0055, ARI 0.0066; M4 0.0 on all six golds; M5 ConEL-2 concept-count range 2.
  "Identity is order-invariant on ConEL-2 and is not on ECB+."

What merge strategy does the resolver use, and does batch differ from streaming?

## Decision Drivers

- Streaming is the primary use case, not a bolt-on (arch :92).
- One code path for batch and streaming.
- Determinism: same inputs give the same bytes (review-findings GC3; stats spec :38).

## Considered Options

- **A. Greedy, arrival-order merging; batch = fold over `process()`.**
- **B. Clustering resolver.** Listed as a possible Resolver adapter (arch :130). Never
  implemented or evaluated.

## Decision Outcome

Chosen option: **A**.

The spec's stated reason is that a single fold keeps streaming first-class. No written
reason for rejecting B, global re-clustering or order-invariant merging exists in the repo
(see Open questions).

**Y-statement:** In the context of a concept graph that accretes over a document stream,
facing the need to decide identity per mention as it arrives, we decided to merge greedily
in arrival order and treat batch as a fold over `process()`, to achieve one streaming-native
code path, accepting that the graph depends on insertion order.

### Consequences

- Good: one code path; streaming and batch can't drift apart.
- Good: deterministic for a fixed order (same sequence gives the same graph, test-enforced
  by save/load resume equivalence).
- Bad: not order-invariant. On ECB+, order spread (b3-f1 0.0037) is 22 % of the
  nn − exact Δ (+0.0171) and exceeds the nn − stemmed CI lower bound (+0.0030).
- Bad: paired-delta intervals hold one insertion order fixed, so order variability is not
  inside the published CIs (README :292-293).
- Bad: duplicate documents bias the greedy pipeline (spec audit STAT-9).
- Test: permutation order-spread report (`23d4a4e`), reported alongside intervals.
- Follow-up: spec audit ARCH-12 (fold order: timestamp vs arrival unspecified), ARCH-7
  ("batch-style" undefined), STAT-8 (glossary-first in permutations?).

## Open questions

- **Scope of "determinism".** Same sequence, same graph (holds today), or same document
  set, same graph (does not hold on ECB+)? This decides what the paper can claim.
- **Embedder determinism across devices and library versions.** Nothing in the repo
  covers it. Default device is CPU (M2 :89-90).
- Rationale not recorded — confirm: why greedy over clustering (arch :130).
- Rationale not recorded — confirm: why first arrival wins, rather than a canonical order.
- Rationale not recorded — confirm: any accepted tolerance for order spread, set in advance.

## What I learned

