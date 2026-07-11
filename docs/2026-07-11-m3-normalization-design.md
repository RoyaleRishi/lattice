# lattice M3 — Normalization Track Design Spec

**Date:** 2026-07-11
**Status:** Approved design, pending spec review
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` (§13 Milestone 3)
**Baseline:** M2b complete at `4981cb2` (210 tests; scorers: frequency / embedding-cosine /
mderank / hcuke; Inspec benchmark harness proven end-to-end)

---

## 1. Goal

Replace the walking skeleton's exact-label resolver with an embedding nearest-neighbour
resolver and produce the project's first credible **normalization** benchmark: clustering
metrics over gold mentions on ECB+ and ConEL-2, comparing `exact-label` vs `embedding-nn`
across a similarity-threshold sweep.

## 2. Decisions log

| Decision | Choice |
|---|---|
| Datasets | **Both** ECB+ and ConEL-2 (user decision 2026-07-11) |
| Evaluation protocol | **Gold mentions** injected via a sidecar extractor; end-to-end extraction+resolution deferred to M5 integration harness |
| Staging | One spec, **one plan** |
| Resolver embedding update | Concept embedding fixed at creation (no centroid updates) — documented deferral |
| Mention embedding | Normalized surface only (consistent with stored concept embeddings); context-aware variants deferred |
| Clustering metrics | B³ (P/R/F1) + ARI, pure Python |
| Literature parity | **Not a criterion** — published CDCR numbers assume gold topic clustering and non-comparable protocols; M3's comparison is internal (embedding-nn vs exact-label) |
| Dataset acquisition | Verified public at spec time (HCUKE lesson): ECB+ via github.com/cltl/ecbPlus; ConEL-2 via github.com/informagi/conversational-entity-linking-2022 |

## 3. Core change (one field)

`GraphDelta` gains:

```python
resolutions: tuple[Resolution, ...] = ()
```

Populated by the orchestrator from the resolver's output (the list already exists at
delta-build time). Backwards-compatible (defaulted). Rationale: clustering evaluation needs
mention → concept assignments; the accreting graph destroys per-document provenance, exactly
as it destroyed ranking provenance before M2's `selected_mentions`. `Resolution` already
carries concept id + scored mention; no other core-type change.

## 4. New adapters

### 4.1 Resolver `"embedding-nn"`

For each selected mention, in input order:

1. `label = mention.surface.strip().lower()`
2. Exact-label short-circuit: if `concept_store.find_by_label(label)` hits, merge with that
   concept (same behaviour as `exact-label`; an identical string must never miss).
3. Else embed the label and query `concept_store.nearest(embedding, k=1)`.
4. If the top hit's cosine similarity ≥ `threshold` → merge: `replace(concept,
   updated_at=document.id)`, upsert, `Resolution(is_new=False)`.
5. Else create a new concept (uuid5 of the label, as `exact-label` does), upsert,
   `Resolution(is_new=True)`.

Params: `threshold: float = 0.8`. Consumes the injected `Embedder` + `ConceptStore` only.
Concept embeddings are **fixed at creation** — no centroid updates in M3 (deferred;
revisit if sweep curves suggest drift matters). One embed batch per document (all labels
in one `embed()` call), then per-mention store queries in order — later mentions in the
same document can resolve to concepts created earlier in that document (stream semantics).

### 4.2 Extractor `"gold-mentions"`

Evaluation-protocol adapter: injects gold mention spans so resolver metrics are not
contaminated by extraction errors (the standard protocol for coreference evaluation).

- Params: `root` (e.g. `data/ecbplus`), `split` (default `"test"`).
- Reads the same converted JSONL the dataset adapter reads (§4.4). Builds
  `{document_id: [mention rows]}` once at construction.
- `extract(units)`: for each unit, look up `unit.document_id`; **assert** the unit's text
  equals the stored document text (guaranteed by the converter's single-newline policy +
  block segmenter → exactly one unit per document); emit `Mention(surface, unit_id,
  span=(start, end), context=...)` per gold row. Mismatch or unknown document id → hard
  error naming the invariant (spec §8: fail loudly).

### 4.3 Scorer `"passthrough"`

`salience = 1.0`, `selected = True` for every mention. The gold-mention protocol must not
drop mentions at the scoring stage. Trivial, deterministic, contract-tested.

### 4.4 Dataset `"mention-clusters"`

**One adapter serves both corpora** — the fetch scripts (§5) converge on one JSONL shape:

```json
{"id": "...", "text": "...", "mentions": [{"start": 0, "end": 5, "surface": "...", "cluster": "chain_7"}]}
```

- Params: `root`, `split` (default `"test"`), `limit` (optional, smoke runs).
- Documents: `kind="article"` (ECB+) / `"transcript"` (ConEL-2) — stored in the JSONL,
  passed through; `timestamp` = stable index order. Text offsets are document-level char
  offsets into `text`; the converter guarantees `text` contains no blank lines (block
  segmenter → one unit).
- `ground_truth()`:
  `{"clusters_by_mention": {f"{doc_id}:{start}-{end}": cluster_id, ...}}` over the split.
- Missing/corrupt data → explicit error naming the fetch script; SHA-256 checksums recorded
  at conversion, verified on load (matches Inspec adapter behaviour).
- Committed 3-document mini-fixtures for each corpus under `tests/fixtures/`.

### 4.5 DocumentMetric `"clustering"`

Cross-document clustering quality over gold mentions:

- Predicted clusters: group every mention key `f"{delta.document_id}:{start}-{end}"` from
  each delta's `resolutions` by `resolution.concept.id`, across **all** deltas
  (`evaluate_documents` receives the full run — cross-document by construction).
- Gold clusters: from `ground_truth["clusters_by_mention"]`.
- A predicted mention key missing from gold, or gold mention missing from predictions →
  **hard error** (eval sets never shrink silently). The gold-mention protocol guarantees
  1:1 coverage; violations mean a broken config, not a metric decision.
- Reports: `"b3-precision"`, `"b3-recall"`, `"b3-f1"` (element-wise B³, macro over
  mentions) and `"ari"` (adjusted Rand index). Both pure Python with hand-computed unit
  tests; no new dependencies.

## 5. Fetch scripts

`scripts/fetch_ecbplus.py` and `scripts/fetch_conel2.py`, each: download from GitHub
(public archives, verified at spec time), convert to the §4.4 JSONL under git-ignored
`data/ecbplus/` and `data/conel2/`, record SHA-256 checksums. Datasets never committed.
Stdlib-only conversion (xml.etree for CROMER; json for ConEL-2).

- **ECB+** (Cybulska & Vossen 2014): CROMER XML from `cltl/ecbPlus`. **Entity chains only**
  (HUMAN/NON_HUMAN participants, LOC, TIME markable types; ACTION/event chains deferred).
  Apply the standard validated-sentences filter
  (`ECBplus_coreference_sentences.csv`). Splits by topic: train 1–35, test 36–45
  (literature standard). Cross-document chains keyed by the corpus `instance_id`.
- **ConEL-2** (Joko & Hasibi 2022): JSON annotations over Wizard-of-Wikipedia
  conversations from `informagi/conversational-entity-linking-2022`. One conversation =
  one document (`kind="transcript"`, turns joined with single newlines, speaker prefixes
  preserved as in the raw data). Mentions: concept + named-entity annotations;
  `cluster` = gold Wikipedia entity id (mentions sharing an entity are one cluster,
  cross-conversation). Personal-entity annotations excluded (they denote speaker-relative
  references, not shared concepts) — documented in the script. ConEL-2 is an evaluation
  collection: it converts as a single `"test"` split (no train split emitted).
- **Plan-time obligation (M2 lesson):** the plan author downloads and inspects both raw
  formats while writing the plan; converter code in the plan is written against real
  files, not memory. Any surprise (schema drift, missing fields) is resolved in the plan,
  not improvised by implementers.

## 6. Configs + sweep

- `configs/m3-ecbplus-sweep.toml` and `configs/m3-conel2-sweep.toml`: base = gold-mentions
  extractor + passthrough scorer + block segmenter + sentence-transformer embedder +
  clustering metric + mention-clusters dataset (respective root); axes:

```toml
[axes]
resolver = [
  { name = "exact-label" },
  { name = "embedding-nn", params = { threshold = 0.65 } },
  { name = "embedding-nn", params = { threshold = 0.75 } },
  { name = "embedding-nn", params = { threshold = 0.80 } },
  { name = "embedding-nn", params = { threshold = 0.85 } },
  { name = "embedding-nn", params = { threshold = 0.90 } },
]
```

- CLI (as built in M2): `python -m lattice.harness --sweep <config> reports/m3-<corpus>`.
- The relation-inducer and snapshot-metric stages run as configured in M1 (co-occurrence,
  in-memory) — unchanged and irrelevant to the clustering metric.

## 7. Error handling

- Gold-mentions extractor: unit/document text mismatch, unknown document id → hard error
  naming the single-unit invariant and the fetch script.
- Clustering metric: mention-key mismatch in either direction → hard error (never shrink).
- Dataset: checksum mismatch aborts; missing files name the fetch script.
- Everything else inherits the orchestrator `on_error` policy unchanged.

## 8. Testing strategy

- Contract suites: `embedding-nn` passes the Resolver contract; `gold-mentions` the
  Extractor contract; `passthrough` the Scorer contract; `mention-clusters` the Dataset
  contract; `clustering` the DocumentMetric contract.
- Pure-unit: B³ and ARI on hand-computed fixtures (including single-cluster and
  all-singletons edge cases); threshold merge/create boundary (≥ vs <); exact-label
  short-circuit; converter span integrity (surface == text[start:end]) on mini-fixtures.
- Integration (ml-marked): mini-fixture end-to-end through `run_experiment` with
  gold-mentions + passthrough + sentence-transformer + embedding-nn; reproducibility
  (identical config → identical report).
- Lean suite stays green without the ml group (hashing embedder paths).

## 9. Success criteria

- Both fetch scripts produce converted JSONL with recorded checksums; converted mention
  spans satisfy `surface == text[start:end]` universally.
- Both sweeps run with **zero errors** and produce stamped JSON + markdown reports.
- `embedding-nn` at some threshold **beats `exact-label` on B³ F1 on at least one
  corpus**; the threshold curve is reported for both.
- Suite green (with and without ml group); ruff clean.
- No new dependencies (stdlib parsing only; embedder reused from M2).

## 10. Explicitly deferred

- Centroid/EMA concept-embedding updates (revisit with M3 curves in hand).
- Context-aware mention embeddings (surface+context windows).
- Event (ACTION) chains in ECB+; personal entities in ConEL-2.
- End-to-end extraction+resolution evaluation (M5 integration harness).
- CEAF/MUC/CoNLL-average metrics (B³+ARI suffice for an internal comparison).
- Persistent ConceptStore backends (parent spec §14).
