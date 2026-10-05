# Spec audit (2026-10-04)

Logic audit of the nine design specs under `docs/`. Two checks only (user's choice):
**internal consistency** and **spec vs spec**. No spec-vs-code comparison. 76 findings came from
nine parallel `system-designer` agents, one per spec. The quotes for every high-severity finding
not already recorded were verified by hand. Findings are grouped into themes by root cause.

How fixes are made (user decision): **design decisions** get an ADR in `decisions/` plus short
dated amendment notes in the specs. **Clarifications** are amended in place with a dated note,
following the repo convention: `(*Amended YYYY-MM-DD, ADR-NNNN: ...*)`.

Line refs in the arch spec are **pre-amendment**. The T1 edits shifted §4 onward by roughly 5–25
lines, so grep for the quoted text.

## Status

| Theme | Status |
|---|---|
| T1 State + error policy | **Done (spec level).** ADR-0001..0003 accepted. Specs amended. Follow-up code done (753b2e9, 0eea93a, ce4129f, 9130d68; C7 pending commit). |
| T2 Selection bias in published comparisons | Not started |
| T3 Interval method (§8 amendment not propagated) | Not started |
| T4 Metric definitions / pass gates | Not started |
| T5 Port contracts silently redefined | Not started |
| T6 Algorithm details underspecified | Not started |
| Unthemed med/low | Not started |

Uncommitted on disk:
- `docs/2026-07-05-lattice-architecture-design.md`
- `docs/2026-07-11-m3-normalization-design.md`
- `docs/2026-07-13-m6-api-hardening-design.md`
- `decisions/0001..0003`
- this file

The user reviews and commits.

## Decisions (T1)

- [ADR-0001](../../decisions/0001-state-ownership-in-the-pipeline.md):
  - `process(document) → GraphDelta`.
  - `Resolver` is a stateful port that **owns** its `ConceptStore` (option B1).
  - Lifecycle `checkpoint/rollback/snapshot/restore`.
  - `ConceptStore` is no longer cross-cutting.
- [ADR-0002](../../decisions/0002-error-policy-and-per-document-atomicity.md):
  - Every document is atomic under both policies (X). The policy only routes the error.
  - `GraphIntegrator` gains `checkpoint/rollback` (i).
  - Checkpoints are undo logs.
  - The policy covers exceptions inside `process()`. Default stays `fail`.
- [ADR-0003](../../decisions/0003-save-format-v2-and-v1-migration.md):
  - Save format v2 adds `resolver_state`.
  - `load` reads only v2.
  - One-off `scripts/migrate_save.py` for v1 files (C).

Spec sections amended:
- arch §4 intro, §4 diagram, §4.1, §4.2, §6, §7.3 (Factory), §8
- M3 §7
- M6 §2 (error-policy row and restore row), §4.3 (v2 paragraph)

"Implementation pending" notes were added to arch §4.2, arch §8 and M6 §4.3. Remove them when
the follow-up code lands. (Removed in C7.)

Findings closed: ARCH-1, 2, 3, 5, 6, 11 · M3-3 · M6-1, 2, 3.

Follow-up code is listed in each ADR's Consequences. All done (commits 753b2e9, 0eea93a,
ce4129f, 9130d68; C7 docs and cleanup pending commit):
- resolver lifecycle
- integrator `checkpoint/rollback`
- undo logs (store + integrator)
- orchestrator checkpoints under both policies, with the `getattr` removed
- save/load v2
- migration script, last

## Open findings by theme

`known` = already recorded in `docs/plans/2026-07-31-review-findings.md`.

### T2 Selection bias in published comparisons (ADR-worthy)
- **STAT-5 (high).** stats §6 L180-181 says "pre-registered operating point nn@0.90 (chosen
  independently in M5)". In fact 0.90 was chosen in M5 on the **same 58 ConEL-2 test
  conversations**, using gold-free metrics. The redundancy evidence it was chosen on was vacuous
  (commit 37e2619). So it is not pre-registered. "Independent" is only half true: no gold labels
  were used.
- **STAT-6 (med).** stats §6 L180-182 silently replaces the M3 §9 L212-213 success criterion
  ("at *some* threshold beats exact-label") with a fixed-threshold criterion. M3 is never
  amended.
- **STAT-7 (high, partly known).** stats §9 L286-288 says "beats = CI excludes 0", but §8
  L269-273 requires family-wise (Holm) control. These are two decision rules.
- **PR-1 (high).** promptrank §6 L157-158 and §5 L137-140 pick the comparison arm as the argmax
  incumbent at f1@15 on the same 500 docs. That conflicts with stats §6 L180-186.
- **PR-2 (high).** promptrank §6 L162-164 allows a "smaller reporting slice", but §2 L43 and §6
  L147, L170 require the same 500 docs.

### T3 Interval method: stats §8 amendment never propagated
- **STAT-1 (high, known).** stats §1 L21-22 and §2 L37 promise BCa everywhere, but §8 L223-230
  uses a subsample band for pooled metrics and percentile-only for holistic ones.
- **STAT-2 (high, known).** §4.1 L93-96 calls the edge-set band a CI, but §8 L243-251 says
  `brackets_estimate=false` means it is not a CI.
- **STAT-3 (high, known).** §3 L66-67 and §9 L282-283 say "interval centered on the published
  number", but §8 L246-249 says bands may not straddle the estimate.
- **STAT-4 (high, known).** §4.2 L122-128 specifies a paired, same-seed, with-replacement
  bootstrap, but §8 requires the pooled subsample path for b3-f1.
- **STAT-9 (med, known).** §4.1 L97-98 resamples holistic metrics with replacement. Duplicate
  documents bias the greedy pipeline.
- **STAT-10 (low).** §5 L168 says the 0.65–0.95 grid "reuses existing points", but the M5 sweep
  has only 3 points.
- **XC-2 (high, known).** cross-corpus §6.3 L106-108 uses a holistic with-replacement bootstrap.
- **XC-3 (med).** cross-corpus §2 L34 sets B=150, but stats §5 L166 and §6 L178 set B=1000. The
  CI table would mix the two.
- **XC-4 (low).** cross-corpus is percentile-only, but stats §2 says BCa (stats §8 agrees with
  cross-corpus).
- **XC-9 (low).** cross-corpus §10 L151-153 runs holistic resampling outside the regime stats §1
  L25-26 calls tractable.
- **PR-7 (med).** promptrank §6 L170-171 copies the M3 `paired_delta` pattern, which is pooled
  subsample. But `f1-at-k` is a macro metric and should use resample.

### T4 Metric definitions / pass gates
- **M2-1 (high).** M2 §6.3 L98-99 sets `top_k` default 10, while §6.5 L123 sets `ks` to
  `[5,10,15]`. So F1@15 is scored on at most 10 predictions.
- **M2-2 (high, known Task 5).** M2 §6.5 L119-123 doesn't say whether to dedupe before or after
  stemming.
- **PR-3 (high).** promptrank §3 L66-71 has the backend return a mean log-prob, while §4 L88-89
  has the adapter length-normalize. That risks double normalization.
- **ARCH-8 (high).** arch §6 and §9 say `Metric` scores a snapshot, but F1@K needs per-document
  rankings and B³ needs a mention→concept map. Neither is in the `Concept` fields (§5 L111).
- **M3-5 (low).** M3 §4.5 L116-117 says "macro over mentions", but M2 §6.5 uses macro to mean
  per-document.
- **M5-3 (med).** M5 §8.2 L189-192 says "strictly decreases", but also says a tie is
  adjudicable.
- **M5-4 (med).** M5 §8.3 L193-194 says "within noise", but each row is a single deterministic
  run.
- **XC-6 (med).** cross-corpus §6.2 L102-104 uses "singleton" with two meanings. The gate fires
  on exact-label by construction.
- **XC-7 (med).** cross-corpus §7 L115-116 requires "monotone", which is stricter than M5 §8.3.
- **XC-8 (med).** cross-corpus compares 58 vs ~200 docs on size-dependent metrics
  (duplicate-rate, singleton-fraction, concept-count).
- **M2-8 (low).** M2 §11 L209-216 sets MDERank's band above cosine's despite the expected
  underperformance.

### T5 Port contracts silently redefined
- **ARCH-4 (med).** "snapshot" means both a port state capture and the graph object metrics
  read. T1 named the lifecycle, but the term is still overloaded.
- **M2-5 (med).** M2 §3 L42 and §7 L156-158 restrict `reset()` to intra-config use, against arch
  §4.2.
- **M2-6 (med).** M2 moves model choice from a scorer `encoder` param (arch §7 example) to the
  embedder.
- **M2-7 (low).** The new `DocumentMetric` port doesn't amend arch §6, §9 or §13.
- **M3-4 (med, partly known T10).** M3 §4.5 L107-111 turns `DocumentMetric` into whole-corpus
  scoring, versus M2 §5 L61-62.
- **M4-6 (low).** M4 §4.4 L128-129 adds a second ground-truth shape, versus M2 §5 L73-74.
- **M5-2 (med).** M5 §7 L169-174 exempts `coherence` from `DocumentMetricContract`, versus arch
  §11 (LSP).
- **M5-5 (low).** M5 §3 L35-43 adds a second injection site in `runner.py`, versus arch §7's
  single composition root. Related: `UnionInducer` calls `lookup` itself
  (`relation_inducer/union.py:17-20`).
- **M5-7 (low).** `Metric` adapters that don't use ground truth, versus arch §6.
- **M6-6 (med).** M6 §3 L36-39 makes `restore` abstract, versus §7 L232 "existing tests pass
  unchanged".

### T6 Algorithm details underspecified
- **M4-1 (high, partly known Task 2).** M4 §4.2 L78-80 defines coordination only for `such as`.
- **M6-5 (high).** M6 §4.1 L83-85 doesn't say whether the auto-id counter advances when the id
  is overridden.
- **M4-2 (med).** M4 §4.5 L139 says longest-match, but L143-145 uses `finditer`, which is
  leftmost-first.
- **M4-3 (med).** M4 §4.2 L83 makes the comma optional, versus the table at L87-90.
- **XC-5 (med).** cross-corpus §2 L32 "first-N in file order" vs §5 L89-90 "sorted by id".
- **M4-5 (low).** `longest_only` has two meanings.
- **M4-8 (low).** "ordered by term" could be alphabetical or file order.
- **ARCH-12 (low).** The fold could follow timestamp order or arrival order.

### Unthemed (med/low)
- **M3-1 (med).** M3 §4.4 L100-102 says checksums are never re-verified, but M3 §7 and M2 §9
  say a mismatch aborts.
- **M3-2 (med).** M3 `limit` vs the metric's coverage hard error.
- **M6-4 (med).** M6 §2 still says "benchmark-validated", which §4.1 retracts.
- **STAT-8 (med).** Do permutation runs hold the glossary first?
- **M4-4 (med).** Does the "participant band" include baseline B?
- **PR-4 (med).** T5 truncation is not asserted.
- **PR-5 (med).** The exclusion reason is recast from M2 §2.
- **PR-6 (med).** "Second model family" has two meanings.
- **ARCH-7 (low).** "batch-style" is undefined.
- **ARCH-9 (low).** "merged" vs stable id.
- **ARCH-10 (low).** "no generative LLM" vs "no generative model".
- **M2-3 (low, known).** Truncate vs assert.
- **M2-4 (low).** Parser disabled vs `noun_chunks`.
- **M3-6 (low).** Explicit amendment.
- **M5-6 (low).** Eager second embedder.
- **M5-8 (low).** Explicit amendment.
- **M6-7 (low).** Adapter change vs "no changes".
- **PR-8 (low).** Generative status settled without amending arch §14.

## Parked

- `Engine.save` writes with a plain `write_text` (`src/lattice/engine.py:~188`), so a crash mid-write can leave a partial save file. This predates ADR-0003; the C6 reviewer flagged it on 2026-10-04.
- `tests/api/test_persistence.py` has no `ml`-marked tests, so save/load is never exercised on the `standard` profile (sentence-transformer embedder). Found by the C5 reviewer, 2026-10-04.
- ~~Finish themes T2–T6 before planning the ADR-0001..0003 follow-up code.~~ Resolved 2026-10-04: the user chose to implement first. The code landed in 753b2e9, 0eea93a, ce4129f and 9130d68, plus C7.
- Code-level risks from the first architecture pass, out of scope for a spec audit:
  - `relations_added` includes edges the graph already had (`orchestrator.py:99`).
  - A repeated edge overwrites the earlier one, losing its provenance and confidence
    (`graph_integrator/in_memory.py:25`).
  - `Concept.updated_at` holds a document id, not a time (`embedding_nn.py:42`).
  - The harness builds a second embedder for metrics (`harness/runner.py:44-46`).

## Next session starts with

1. Read this file and `decisions/0001..0003`. T1 is fully done: specs amended and code implemented
   (see `docs/plans/2026-10-04-adr-0001-0003-implementation.md`).
2. Start **T2** (selection bias). STAT-5 and PR-1 are design decisions, so they go the ADR route:
   present options and trade-offs, and the user decides.
3. Then T3 → T6 → unthemed, one theme at a time.
