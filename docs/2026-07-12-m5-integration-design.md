# lattice M5 — Integration Harness Design Spec

**Date:** 2026-07-12
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` §9/§13 milestone 5
**Status:** approved design, pre-plan

Milestone 5: the bespoke **intrinsic** harness — redundancy rate, cluster
purity (as concept coherence), hierarchy sanity — judged over the accreting
graph produced by the FULL real pipeline streaming a small transcript
corpus. First milestone where no stage runs on gold crutches: real
extraction, real salience, real resolution, real induction, one graph.

## 1. Goal

Answer "is the accreted graph any good?" without gold labels. The two merge
metrics are designed as a tension pair: **redundancy** measures what the
resolver failed to merge; **coherence** measures what it wrongly merged.
Moving the merge threshold trades one for the other, and the harness's job
is to expose that trade-off so an operating point can be chosen
intrinsically — plus structural sanity checks on the induced IS_A hierarchy.

## 2. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Corpus | ConEL-2 test split (58 conversations, on disk from M3) | Closest to the LLM-session framing; zero new fetch/convert work; gold clusters available for a recorded qualitative cross-check without becoming the metric. User decision 2026-07-12. |
| Harness shape | Three separate metric adapters behind existing ports | Parent spec: "metrics are adapters … not separate scaffolding". Rejected: monolithic intrinsic-suite metric (SRP/ISP violation); standalone analysis script (explicitly against parent spec §9). |
| Embedder access for metrics | Runner instantiates a second embedder from `config.embedder` and offers it as a shared dep to metric construction | Only `coherence` consumes it. `build_orchestrator` keeps its signature; the duplicate model load (~100 MB, deterministic same embeddings) is accepted. Injection reuses `instantiate`'s existing param-name matching, so dep-free metrics are unaffected. |
| Gold cross-check | Qualitative, ledger-recorded only | Real noun-chunk mentions don't align with gold spans, so M3's clustering metric cannot apply (its coverage gate would rightly refuse). Top-concepts inspection against gold entities is recorded in the ledger; never an exit criterion. |
| Extraction realism vs test purity | Real run uses spaCy noun-chunk + sentence-transformer (ml group, models already cached); tests use token extractor + hashing embedder | No models or network in tests (standing constraint); the ml path gets one `@pytest.mark.ml` e2e, skipped when models are absent (M3 pattern). |
| New dependencies | None | Tarjan SCC and longest-path are ~40 lines of stdlib. `pyproject.toml` stays frozen. |

## 3. Core changes

**None in `core/` or `ports/`.** One harness change in
`src/lattice/harness/runner.py`: metric and document-metric instantiation
gains shared-dep injection —

```python
metric_shared = {"embedder": instantiate(Embedder, config.embedder)}
...
instantiate(Metric, spec, metric_shared)
instantiate(DocumentMetric, spec, metric_shared)
```

`instantiate` already injects only when the constructor names the param, so
existing metrics (`label-f1`, `edge-f1`, `f1-at-k`, `clustering`) are
untouched. The second embedder instance is constructed lazily-enough (once
per experiment) and is deterministic, so pipeline and metric embeddings
agree.

## 4. New adapters

### 4.1 Metric `"redundancy"` (snapshot-level)

Constructor: `(threshold: float = 0.9)` — no shared deps; reads stored
`Concept.embedding` from the snapshot.

Two concepts are **near-duplicates** when either:
- cosine(embedding_a, embedding_b) ≥ `threshold` (both non-zero vectors), or
- `normalize(label_a) == normalize(label_b)`, where `normalize` =
  casefold → strip one leading article ("a ", "an ", "the ") → strip one
  trailing "s" when the result stays ≥ 3 chars and the label does not end
  in "ss" ("beatles"→"beatle", "glass"→"glass").

O(n²) pairwise scan (fine at this scale; ConEL-2 test yields well under
5,000 concepts). Returns:
- `duplicate-rate`: |{c : ∃c′≠c near-duplicate of c}| / |C| (0.0 when the
  snapshot has no concepts),
- `near-duplicate-pairs`: count of unordered near-duplicate pairs (float),
- `concept-count`: |C| (float).

### 4.2 DocumentMetric `"coherence"`

Constructor: `(embedder: Embedder)` — injected via §3.

Fold over the run's deltas: group **distinct casefolded mention surfaces**
by `resolution.concept.id` across all resolutions. For each concept with
≥ 2 distinct surfaces, coherence(c) = mean pairwise cosine over one batched
`embedder.embed(sorted(surfaces))` call (all distinct surfaces across all
concepts go in a single batch; zero vectors contribute 0 to their pairs).
Returns:
- `coherence`: mean over multi-surface concepts; **1.0 when there are none**
  (nothing was incoherently merged — vacuous perfection made visible by:),
- `multi-surface-concepts`: count (float),
- `singleton-fraction`: fraction of concepts with exactly one resolution
  across the whole run (fragmentation signal; 0.0 when no concepts).

### 4.3 Metric `"hierarchy-sanity"` (snapshot-level)

Constructor: `()` — no deps. Operates on the snapshot's `type == "IS_A"`
edges only (deduped by construction in the integrator). All stdlib:

- `cycle-components`: number of strongly connected components of size ≥ 2
  (iterative Tarjan; recursion-free so deep chains can't blow the stack),
- `cycle-nodes`: total nodes across those components,
- `self-loops`: count of edges with source == target (construction should
  make this 0; counted so a regression is visible),
- `max-depth`: longest path (in edges) over the subgraph excluding
  cycle-component nodes and self-loops, via memoized iterative DFS; 0.0
  when there are no IS_A edges,
- `transitive-shortcuts`: count of edges (a, c) where c is reachable from a
  through a path of ≥ 2 IS_A edges not using the edge (a, c) itself,
- `is-a-edges`: count (float).

All values floats (Metric convention).

## 5. The run

No new dataset or fetch work: dataset `mention-clusters` with
`root = "data/conel2"`, `split = "test"` (58 transcripts; `kind:
"transcript"` documents already on disk, checksums recorded in M3).

Pipeline row (the first all-real configuration in the project):

| port | adapter |
|---|---|
| segmenter | `block` (M3 precedent: converted transcripts are single-newline joined — one unit per conversation) |
| extractor | `noun-chunk` (spaCy `en_core_web_sm`, cached by `scripts/fetch_models.py`) |
| scorer | `embedding-cosine` (adapter defaults) |
| resolver | **sweep axis** |
| relation_inducer | `union` of `hearst` + `compound` (M4) |
| graph_integrator | `in-memory` |
| embedder | `sentence-transformer` (cached) |
| concept_store | `in-memory` |

`configs/m5-conel2-sweep.toml`, axes:

```toml
[axes]
resolver = [
  { name = "exact-label" },
  { name = "embedding-nn", params = { threshold = 0.90 } },
  { name = "embedding-nn", params = { threshold = 0.75 } },
  { name = "embedding-nn", params = { threshold = 0.65 } },
]
```

4 rows; metrics `redundancy` + `hierarchy-sanity` under `[[base.metrics]]`,
`coherence` under `[[base.document_metrics]]`. Run:
`uv run --no-sync python -m lattice.harness --sweep
configs/m5-conel2-sweep.toml reports/m5-conel2`.

## 6. Error handling

- Missing ConEL-2 data → existing `mention-clusters` FileNotFoundError
  (names the fetch script).
- Missing models → existing OSError path; the ml e2e test skips with the
  `fetch_models.py` hint (M3 pattern).
- Metrics never raise on degenerate input: empty snapshot / no deltas /
  no IS_A edges return the documented zero/vacuous values (contract-tested).
- `on_error = "fail"` in the sweep, as always.

## 7. Testing strategy

- **Unit, redundancy:** planted near-duplicate pair by embedding (two
  hand-built unit vectors above/below threshold), planted label collision
  ("the beatles" vs "beatle"), "glass"/"ss" guard, empty snapshot,
  all-distinct snapshot → 0.0.
- **Unit, coherence:** hashing embedder; a concept merging "beatles" and
  "the beatles" (high pairwise cosine) vs one merging "beatles" and
  "kid rock" (low); vacuous 1.0 when no multi-surface concepts;
  singleton-fraction arithmetic; single batched embed call asserted via a
  counting fake embedder.
- **Unit, hierarchy-sanity:** planted 2-cycle a→b→a (1 component, 2 nodes),
  planted self-loop, chain a→b→c (max-depth 2), planted shortcut a→c beside
  a→b→c (1 shortcut), non-IS_A edges ignored, empty snapshot all-zeros.
- **Contract:** `redundancy` and `hierarchy-sanity` join `MetricContract`;
  `coherence` joins `DocumentMetricContract`.
- **Runner injection:** a fake metric with an `embedder` constructor param
  receives one; `label-f1` (no param) still instantiates — both via a tiny
  experiment over the toy dataset.
- **e2e (pure):** full pipeline over `tests/fixtures/mini_clusters_conel`
  with token extractor + hashing embedder + union inducer; all three
  metrics present in the report, zero errors, reproducible run.
- **e2e (`@pytest.mark.ml`):** noun-chunk + sentence-transformer over the
  same fixture; skips when models are absent.

## 8. Success criteria

1. **Mechanical:** all 4 sweep rows complete with zero errors; full suite +
   ruff clean; nothing new committed under `data/` or `reports/`.
2. **Discrimination (hard):** `duplicate-rate` is highest for `exact-label`
   and lower for every `embedding-nn` row; `concept-count` strictly
   decreases along exact-label → nn@0.90 → nn@0.75 → nn@0.65 (an exact tie
   between adjacent nn rows is adjudicable in the ledger; exact-label must
   strictly exceed all nn rows).
3. **Tension (expected, adjudicable):** `coherence` does not increase as
   the threshold loosens (nn@0.65 ≤ nn@0.90 within noise). Intrinsic
   metrics are themselves the deliverable: a violated expectation with a
   recorded, evidence-backed explanation is a finding, not automatically a
   failure — but it must be adjudicated in the ledger before close.
4. **Hierarchy:** `self-loops` == 0 on every row; `is-a-edges` > 0 (the
   union inducer must actually fire on real transcripts); cycle and
   shortcut counts recorded and inspected.
5. **Qualitative cross-check (recorded, non-gating):** top-10 concepts by
   resolution count for the chosen operating row, with their member
   surfaces, eyeballed against ConEL-2 gold entities; note in the ledger.

## 9. Explicitly deferred

- Gold-based purity metrics (extrinsic clustering already covered by M3 on
  gold mentions; span-aligning real extraction to gold is its own project).
- Persistent `GraphIntegrator`/`ConceptStore` backends (parent spec §14).
- Additional corpora (MediaSum etc.) and cross-domain intrinsic profiles.
- Redundancy auto-repair (merging near-duplicates post-hoc) — M5 measures;
  acting on measurements is engine work, not harness work.
- M6 engine-API hardening.
