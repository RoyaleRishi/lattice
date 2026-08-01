# Implementation plan — 2026-07-31 code-review findings

**Branch:** `fix/review-findings`
**Origin:** a full code-level review of lattice at `d9cfe93`. Every finding
below was confirmed by executing code, not read off documentation.

## Context

lattice is a concept-memory engine: documents in, an accreting normalized
concept graph out. Hexagonal architecture — ports in `src/lattice/ports/`,
adapters in `src/lattice/adapters/`, one composition root in
`src/lattice/config/factory.py`, a thin orchestrator in
`src/lattice/orchestrator/orchestrator.py`.

The review found that three of the four README headline claims do not
survive scrutiny, and that two published baselines are affected by
implementation defects. The tasks below fix the defects, then regenerate the
evidence, then correct the README. Ordering matters: the README task is last
because it depends on numbers the earlier tasks change.

## Global Constraints

- **GC1 — Ports stay stable.** Fixes go in adapters or the core; do not
  change a port's abstract signature unless a task explicitly says to.
- **GC2 — Every adapter satisfies its port contract.** New adapters MUST
  inherit the matching contract class from `tests/contracts/` (e.g. a new
  Resolver test inherits `ResolverContract`). This is how the project
  enforces Liskov substitutability; it is not optional.
- **GC3 — Determinism is a hard requirement.** Sorted iteration, seeded
  RNGs, lexicographic tie-breaks. Identical runs produce identical bytes.
  Never introduce set-iteration order or unsorted dict iteration into
  anything that reaches output.
- **GC4 — Honest failure over silent degradation.** Prefer raising to
  returning a plausible-looking wrong number. This is an existing project
  principle (see `gold_mentions.py`, `clustering.py` coverage checks).
- **GC5 — Document deviations from published algorithms** in the adapter
  docstring, in the existing "documented deviations" style.
- **GC6 — Run `.venv/bin/ruff check .` before committing.** It currently
  passes clean; keep it that way. Line length 100.
- **GC7 — Tests run with `PYTHONPATH=src .venv/bin/python -m pytest`.**
  The venv's editable-install `.pth` file is flagged `UF_HIDDEN` on this
  machine and Python 3.13 skips it, so the package is not importable
  without `PYTHONPATH=src`. This is an environment quirk, not a project
  bug — do not try to "fix" it by editing packaging.
- **GC8 — Do not change published numbers by hand.** Any number in a doc
  must come from a regenerated report file.
- **GC9 — No new required runtime dependencies.** Core deps are `pydantic`
  and `snowballstemmer` only. Anything heavier goes in the `ml` extra.

## Verification baseline

Before starting: `PYTHONPATH=src .venv/bin/python -m pytest -q` → 460 passed.
Every task must leave the suite green.

---

## Task 1 — Numerical safety in `cosine`, and dimension validation on load

**Files:** `src/lattice/core/vectors.py`, `src/lattice/engine.py`,
`tests/core/test_vectors.py`, `tests/api/test_persistence.py`

`cosine()` uses `zip(a, b)` without `strict=True`. A dimension mismatch is
silently truncated to the shorter vector while both full norms are used in
the denominator, producing a plausible number instead of an error. The
reachable path is `Engine.load`, which restores stored embeddings into a
live store without ever checking them against the embedder's `dim`.

**Required changes:**

1. In `src/lattice/core/vectors.py`, change `zip(a, b)` to
   `zip(a, b, strict=True)`. Update the docstring to state that
   mismatched dimensions raise `ValueError`.
2. In `Engine.load` (`src/lattice/engine.py`), after building the
   `concepts` tuple and before restoring it, validate every concept
   embedding against the rebuilt engine's embedder dimension. Reach the
   embedder via `engine._orchestrator.resolver.embedder` guarded with
   `getattr(..., "embedder", None)`; skip validation if absent. On
   mismatch raise `ValueError` naming the concept id, the found length,
   and the expected `dim`.

**Tests:**

- `cosine` with vectors of different length raises `ValueError`.
- Existing zero-vector behaviour (returns `0.0`) is unchanged — there is
  already a test for this; do not alter it.
- A save file whose concept embedding has the wrong dimension makes
  `Engine.load` raise `ValueError` mentioning the concept id. Build the
  fixture by saving a real lite-profile engine, editing the JSON to
  truncate one embedding, and writing it back.

---

## Task 2 — Hearst: coordinate hyponym lists that precede the hypernym

**Files:** `src/lattice/adapters/relation_inducer/hearst.py`,
`tests/adapters/test_hearst_inducer.py`

`_walk_coordination` is only invoked when `pattern.hyper == "left"`
(hearst.py:109-112). Hearst (1992) patterns 4 and 5 — `and other` /
`or other` — put the hyponym list *before* the hypernym, so only the last
hyponym in the list receives an edge.

Confirmed behaviour today:

```
"fats such as olive oil, canola and margarine"     -> 3/3 edges (correct)
"bruises, wounds, broken bones and other injuries" -> 1/3 edges (only the last)
```

**Required changes:**

Add a backward coordination walk for `hyper == "right"` patterns. After
pairing `(left, right)`, walk *backwards* from the left anchor through
preceding anchors while each gap matches `_COORD` exactly, emitting
`(that anchor, right)` for each. Mirror the existing
`_walk_coordination` structure: stop at the first gap that does not
`fullmatch`, and stop on overlapping spans.

Set `coordination=True` on the `and-other` and `or-other` patterns.

**Tests:**

- `"bruises, wounds, broken bones and other injuries"` with anchors
  `["bruises","wounds","broken bones","injuries"]` yields exactly the
  three edges `(bruises,injuries)`, `(wounds,injuries)`,
  `(broken bones,injuries)`.
- `"olive oil and other fats"` still yields exactly `(olive oil, fats)`.
- Forward coordination is unregressed:
  `"fats such as olive oil, canola and margarine"` → 3 edges.
- A non-coordination gap breaks the walk: in
  `"bruises, we saw wounds and other injuries"` (anchors `bruises`,
  `wounds`, `injuries`) `bruises` does NOT get an edge.

---

## Task 3 — `on_error="skip"` must not diverge the concept store from the graph

**Files:** `src/lattice/orchestrator/orchestrator.py`,
`src/lattice/engine.py`, `tests/orchestrator/test_orchestrator.py`,
`tests/api/test_persistence.py`

Under `on_error="skip"`, the resolver upserts concepts into the
`ConceptStore` before later stages run. If a later stage (relation
inducer, graph integrator) raises, those concepts stay in the store but
never reach the graph. Confirmed consequences:

- The store holds concepts the graph has never seen.
- A later document that re-mentions such a concept pulls it into the
  graph carrying `first_seen` pointing at the document that FAILED and
  contributed nothing.
- Resume-equivalence breaks: straight-through gives `('doc-1','doc-2')`
  for that concept while save/load/resume gives `('doc-2','doc-2')`.
  The README calls resume-equivalence test-enforced.

The reason this went unnoticed: the only `skip` test uses an
`ExplodingExtractor`, which fails *before* the resolver mutates anything.

**Required changes:**

Make the skip path roll the concept store back to its pre-document state.
In `Orchestrator.process`, when `on_error == "skip"` and an exception is
caught, restore the store. Get the store the same way `Engine` does:
`getattr(self.resolver, "concept_store", None)`.

Add two methods to the `ConceptStore` port
(`src/lattice/ports/concept_store.py`) and implement them in
`InMemoryConceptStore`:

- `checkpoint() -> object` — return an opaque snapshot token of current
  state.
- `rollback(token: object) -> None` — restore state to that token.

For `InMemoryConceptStore`, the token is a shallow copy of both internal
dicts (`dict(self._by_id)`, `dict(self._id_by_label)`); `Concept` is a
frozen dataclass so a shallow copy is sufficient.

`Orchestrator.process` takes a checkpoint before the try block (only when
`on_error == "skip"` and a store is reachable) and rolls back in the
except branch before returning the error delta. Update the
`Orchestrator` class docstring: the partial-mutation caveat it currently
documents is now fixed for the concept store.

Also correct the now-stale premise in `Engine.save`'s docstring, which
says "resolvers upsert exactly the concepts the integrator holds" — after
this task that is true; make the docstring say why (the skip path rolls
back).

**Tests:**

- A stage that fails AFTER the resolver (fail in the relation inducer)
  under `on_error="skip"` leaves the concept store byte-identical to its
  pre-document state. Assert on store contents, not just the delta.
- `ConceptStoreContract` gains checkpoint/rollback coverage: after
  `checkpoint()`, an `upsert`, then `rollback()`, the store returns the
  original contents.
- Resume-equivalence holds under `on_error="skip"` when a document fails
  mid-pipeline. Use this exact scenario, which fails today:
  - config: block segmenter, token extractor, embedding-cosine scorer,
    embedding-nn resolver at threshold 0.99, hearst inducer, in-memory
    integrator, hashing embedder, `on_error="skip"`
  - A = `"Alpha discusses kubernetes orchestration."`
  - B = `"Beta mentions kubernetes again."` (force the relation inducer
    to raise on this document only)
  - C = `"Gamma revisits kubernetes deployments again."`
  - Assert `ingest(A,B,C)` produces the same snapshot as
    `ingest(A,B); save; load; ingest(C)`.

---

## Task 4 — HCUKE: normalize significance components before the Eq. 7 product

**Files:** `src/lattice/adapters/scorer/hcuke.py`,
`tests/adapters/test_hcuke_scorer.py`

The reference paper is at `docs/papers/hcuke-knosys-2024.pdf` (gitignored
but present locally). Published Inspec F1@5/10/15 = 38.34 / 43.41 / 42.79.
lattice currently reports 15.07 / 17.34 / 19.82 — below the trivial
frequency baseline, making it a strawman rather than a beaten competitor.

Root cause, confirmed. `hcuke.py:97-104` computes
`local_sig[s] = Σ_other (sim(s,other) − λ·μ)` = `n·mean_sim_s − λ·n·μ`.
With λ=1.3 and `mean_sim_s` averaging to ≈ μ, this is a constant negative
shift. Eq. 7 is a *product*, so a negative local term inverts the ranking
with respect to global significance. Measured on 60 Inspec test documents:
**1094/1595 (68.6%) of candidates get negative salience**, and in
**50/60 documents the majority are negative**.

Measured, same code, λ set to 0.0: F1@5/10/15 = 28.88 / 35.49 / 36.68.
The paper's Figure 3 shows F1@10 varying by **less than one point** as λ
sweeps 0 → 1.5; here it swings seventeen points. Something the paper does
is missing from this implementation.

> **Correction (applied during execution).** An earlier draft of this task
> asserted that §3.4 mandates normalizing the Eq. (7) factors. It does not.
> §3.4 says only "simple filtering and normalization operations", and
> Algorithm 1 line 19 forms the product from the raw factors. The fix below
> is still correct, but its justification is **internal** — a product with a
> negative factor inverts the ranking with respect to the other factors —
> and the adapter docstring must not credit the paper for it.

**Required changes:**

1. **Normalize the unbounded factors before multiplying.** Min-max normalize
   `global_sig` and `local_sig` independently to [0, 1] across the
   document's candidate set before computing the product. Guard the
   degenerate case (all values equal → range of 0): map every value to 1.0
   rather than dividing by zero.

   **Leave `candidate_weight` (W(c)) raw.** It comes out of `_softmax`, so
   it is already a normalized distribution; stretching a softmax to span
   [0, 1] is distortion, not normalization, and it inflates a signal the
   paper deliberately keeps weak. Document the asymmetry as deliberate.
2. **Compute μ over all n² ordered pairs including the diagonal**, per
   §3.4: `μ = (1/n)Σᵢ(1/n)Σⱼ dist(H_cᵢ, H_cⱼ)`. The current code averages
   unique off-diagonal pairs only (`hcuke.py:91-96`). Keep the existing
   `1.0` self-similarity shortcut rather than calling `cosine(v, v)` —
   it avoids float drift and is already the established choice.
3. Extend the docstring's documented-deviations ledger with the
   normalization step and the μ definition, and keep the existing
   Eq.(5)-vs-Algorithm-1 note (that reading is correct — do not change it).

**Do not change** the decontextualized-representation deviation (whole-string
MiniLM vs the paper's max-pooled contextual BERT). It is real, already
documented, and out of scope. It is the dominant known contributor to the
residual gap against the published numbers, but its size is **not**
established — an earlier draft of this plan attributed ~8 F1 points to the
paper's Table 3, and that attribution is wrong (the Table 3 Inspec F1@10
spread is ~3.3 points). Do not quote a figure for it anywhere, here or in
`docs/results`.

**Tests:**

- The existing hand-computed fixture test
  (`test_hcuke_scorer.py:41-73`) will need its expected values recomputed
  — it currently asserts a negative `R_l(gamma) = −0.3`, which is the
  pathology. Recompute every intermediate by hand for the new formula and
  assert the new values. Show the arithmetic in comments.
- A new test asserting the invariant that motivated the fix: for a
  document with ≥3 distinct candidates, every returned salience is
  `>= 0.0`. This is what normalization buys, and it is what regressed.
- Keep `ScorerContract` inheritance.

**Acceptance:** after the fix, HCUKE on 60 Inspec test documents must
score F1@10 substantially above the frequency baseline's 26.00. Report the
measured F1@5/10/15 in your report file. Use this snippet shape to measure
(it is how the numbers above were produced):

```
cfg = load_config("configs/m2b-sweep.toml", model=SweepConfig).base
cfg.scorer = type(cfg.scorer)(name="hcuke", params={"top_k": 15})
docs = list(instantiate(Dataset, cfg.dataset).documents())[:60]
deltas = build_orchestrator(cfg).process_stream(docs)
instantiate(DocumentMetric, cfg.document_metrics[0]).evaluate_documents(deltas, gt)
```

---

## Task 5 — F1@K: use the literature's precision denominator

**Files:** `src/lattice/adapters/document_metric/f1_at_k.py`,
`tests/adapters/test_f1_at_k.py`

`f1_at_k.py:35-54` dedupes by *raw surface*, truncates to k, then builds a
**set of stems** and divides precision by `len(predicted)`. When two of the
top-k surfaces stem alike ("neural network" / "neural networks"), a slot is
consumed but the denominator shrinks — inflating precision.

The literature protocol (and the HCUKE paper §4.1 is explicit about it)
dedupes *first*, then takes the top k, keeping the denominator at k.
Measured impact: 10/40 documents lose slots at k=15; precision@15 moves
33.91 → 32.40 for the cosine baseline. It does not reorder systems, but it
means the reported absolutes are not comparable to the published bands the
project uses as sanity checks.

**Required changes:**

Dedupe on the **stemmed** phrase while ranking, keeping the highest-salience
surface for each stem, then truncate to k. Precision denominator becomes
`len(ranked_stems[:k])`, which equals `min(k, n_distinct_stems)`.

Concretely, change `_ranked_unique_surfaces` to return ranked *stems*:
key `best` by `self._stem_phrase(surface)` instead of raw surface,
keeping max salience per stem, and preserve the existing
`(-salience, key)` tie-break. `_per_document_scores` then compares
`set(ranked[:k])` against the stemmed gold set directly, with no second
stemming pass.

Update the class docstring to state the dedupe-then-truncate ordering and
why (comparability with published tables).

**Tests:**

- A document whose top-k surfaces contain two that stem alike consumes
  only one slot, and precision@k divides by the deduped count. Construct
  the fixture so the old and new behaviour differ, and assert the new
  number with the arithmetic shown.
- Existing tests that encode the old denominator must be updated, not
  deleted — recompute their expected values.
- Keep `DocumentMetricContract` inheritance.

---

## Task 6 — MDERank: detect truncation instead of silently unranking candidates

**Files:** `src/lattice/adapters/scorer/mderank.py`,
`tests/adapters/test_mderank_scorer.py`

`mderank.py:20-23` asserts "Inspec abstracts (~122 words) fit MiniLM's
256-token window, so no truncation handling". That claim is false:
61/500 Inspec test abstracts exceed 256 word-pieces (median 158, max 497).
Masking beyond the cutoff is a no-op, so `cos = 1` and `1 − cos ≈ 0`, and
those candidates are silently unrankable.

> **Correction (applied during execution).** An earlier draft of this task
> said the longest document had 22 such candidates scoring "exactly
> 2.22e-16". Re-measured against the real Inspec split and the real MiniLM
> tokenizer: the longest document is id `392`, it yields **60** distinct
> candidate surfaces of which **28** are degenerate, the mean salience of
> the non-degenerate remainder is **0.0242**, and the degenerate values are
> **−2.22e-16** (negative). The `61/500`, median `158` and max `497` figures
> are confirmed correct. Use only the re-measured numbers.

**Required changes:**

Detect the condition rather than guess at the tokenizer. A candidate whose
masked-document embedding is (near-)identical to the unmasked document
embedding contributed nothing to the ablation. After computing `salience`,
count surfaces whose value is `<= 1e-12`, and if any exist, record it.

Add a public attribute `self.degenerate_surfaces: int` reset per `score()`
call, holding that count. Do not raise — a partially-truncated document
still yields a usable ranking for its early candidates, and raising would
break existing benchmark runs (GC4 favours honest failure, but here the
honest signal is a count the harness can surface, not a crash).

Correct the docstring: state that documents exceeding the embedder's
context window yield near-zero salience for late candidates, that this is
detected via `degenerate_surfaces`, and drop the false "no truncation
handling needed" claim.

**Tests:**

- A stub embedder that returns an identical vector regardless of input
  makes every surface degenerate; assert `degenerate_surfaces` equals the
  surface count and that `score()` still returns one `ScoredMention` per
  input mention.
- A normal document leaves `degenerate_surfaces == 0`.
- `degenerate_surfaces` resets between calls.
- Keep `ScorerContract` inheritance.

---

## Task 7 — Add a `stemmed-label` resolver as an honest identity baseline

**Files:** new `src/lattice/adapters/resolver/stemmed_label.py`,
`src/lattice/adapters/resolver/__init__.py`,
new `tests/adapters/test_stemmed_label_resolver.py`,
new `configs/m3-conel2-stemmed.toml`, new `configs/m3-ecbplus-stemmed.toml`

The review measured what `embedding-nn` at the shipped 0.90 threshold
actually contributes over `exact-label` on ConEL-2: **7 merges out of 452
resolutions (1.55%)**, every one of them a morphological or spelling
variant —

```
'oceans'->'ocean'  'vegetable'->'vegetables'  'color'->'colors'
'sunflower'->'sunflowers'  'online game'->'online games'
'cartoon networks'->'cartoon network'  'favourite colour'->'favorite color'
```

The Snowball stemmer — already a **core dependency**, already used in
`f1_at_k.py` — reproduces 6 of those 7. So the honest baseline for the
identity claim is not `exact-label`; it is stemmed-label matching. This
task adds it so the comparison can be made.

**Required changes:**

Create `StemmedLabelResolver`, registered as `@register(Resolver,
"stemmed-label")`. It is `ExactLabelResolver` with one change: the merge
key is the Snowball-stemmed label rather than the raw lowercased label.
Model it closely on `src/lattice/adapters/resolver/exact_label.py`.

- Constructor: `(self, embedder: Embedder, concept_store: ConceptStore)`,
  matching `ExactLabelResolver` so the factory injects both.
- Stem with `snowballstemmer.stemmer("english")`, joining stemmed
  whitespace tokens — use exactly the same normalization as
  `f1_at_k.py:32-33` so the two agree.
- The **stored `Concept.label` is the first surface form seen**, not the
  stem — the stem is a matching key, not a display label. Keep
  `Concept.id` derived from the stem via
  `uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{stem}")` so
  identity is stable.
- Look up by stem. `ConceptStore.find_by_label` is keyed on
  `Concept.label`, which is the surface form — so it cannot be used for
  stem lookup. Keep the resolver's own `dict[str, str]` mapping stem →
  concept id and resolve through `concept_store.get(...)`. Document that
  this mapping is resolver-local state.
- Batch the embedder call over unique labels, as `embedding_nn.py:31-33`
  does — do not embed one at a time.

Add the two sweep configs by copying `configs/m3-conel2-sweep.toml` and
`configs/m3-ecbplus-sweep.toml` and replacing the `[axes] resolver` list
with exactly three entries: `exact-label`, `stemmed-label`, and
`embedding-nn` at `threshold = 0.90`. This is the three-way comparison the
README's identity claim needs.

**Tests:**

- New test file inherits `ResolverContract` (GC2).
- `"oceans"` and `"ocean"` in different documents resolve to one concept.
- `"favourite colour"` and `"favorite color"` do NOT merge (Snowball does
  not normalize British/American spelling) — this documents the known
  1-of-7 gap and prevents an overclaim.
- The stored label is the first surface seen, not the stem: after
  ingesting `"vegetables"` then `"vegetable"`, the concept's label is
  `"vegetables"`.
- Distinct concepts stay distinct: `"vector store"` and
  `"vector database"` do not merge.

---

## Task 8 — Decouple the redundancy metric from the resolver threshold

**Files:** `src/lattice/adapters/metric/redundancy.py`,
`configs/m5-conel2-nn090.toml`, `configs/m5-ecbplus-nn090.toml`,
`configs/m5-multiwoz-nn090.toml`, `configs/m5-conel2-sweep.toml`,
`configs/m5-ecbplus-sweep.toml`, `configs/m5-multiwoz-sweep.toml`,
`tests/adapters/test_redundancy_metric.py`

The redundancy metric flags concept pairs at `cosine >= threshold`
(default 0.9). The `embedding-nn` resolver creates a concept only when its
nearest neighbour is *below* its own threshold. When the two thresholds
are equal — which is exactly the shipped M5 configuration — no pair can
possibly be flagged. Confirmed on the real M5 ConEL-2 run:

```
concepts: 523
near-duplicate pairs by COSINE only : 0     <- forced by construction
near-duplicate pairs by LABEL only  : 4     ('choirs'/'choir', 'singer'/'singers',
                                             'boxer'/'boxers', 'songs'/'song')
```

So the README's "duplicate-rate 0.117 → 0.015" is a theorem, not a
measurement, and its entire residual is plural/singular label collisions.

**Required changes:**

1. Make the circularity impossible to configure by accident. Add to
   `RedundancyMetric.__init__` no new parameter — instead, split the
   reported output so the two criteria are separable:
   `duplicate-rate`, `near-duplicate-pairs`, `concept-count` (all
   existing, unchanged semantics) plus two new keys
   `cosine-duplicate-pairs` and `label-duplicate-pairs` giving the
   decomposition. A pair meeting both criteria counts in both new keys
   and once in `near-duplicate-pairs`.
2. Extend the class docstring with the circularity warning: when this
   metric's `threshold` is >= the `embedding-nn` resolver's threshold,
   the cosine criterion is vacuous by construction, and only the label
   criterion carries signal. State that configs should set the metric
   threshold **below** the resolver's.
3. Update the six M5 configs to set the redundancy metric threshold to
   `0.80` where the resolver runs at 0.90, so the metric measures
   something the resolver did not already guarantee. The sweep configs
   vary the resolver threshold across `{0.90, 0.75, 0.65}` — for those,
   set the metric threshold to `0.60`, below every point on the axis.
   Add a comment in each config saying why the value is what it is.

**Tests:**

- Two concepts at cosine >= threshold with different labels increment
  `cosine-duplicate-pairs` and not `label-duplicate-pairs`.
- Two concepts with colliding normalized labels but low cosine increment
  `label-duplicate-pairs` and not `cosine-duplicate-pairs`.
- A pair meeting both increments both new keys, and
  `near-duplicate-pairs` counts it once.
- Existing tests keep passing (the three original keys are unchanged).
- Keep `MetricContract` inheritance.

---

## Task 9 — Remove write-only fields

**Files:** `src/lattice/core/types.py`,
`src/lattice/adapters/extractor/token.py`,
`src/lattice/adapters/extractor/noun_chunk.py`,
`tests/core/test_types.py`, plus any test referencing the removed fields

Three fields are written and never read anywhere in `src/`:

- `Unit.speaker` — never populated by any segmenter; `kind="turn"` is
  documented in the `Unit` docstring but no segmenter produces it.
- `Mention.head` — set by `token.py:36` and `noun_chunk.py:41`, read by
  nothing.
- `Mention.lemma` — set by `token.py:37` and `noun_chunk.py:42`, read by
  nothing. The noun-chunk extractor pays spaCy lemmatization
  (`" ".join(t.lemma_ for t in tokens)`) for a field no one consumes.

**Required changes:**

Delete `Mention.head` and `Mention.lemma` from the dataclass and remove
every assignment site, including the spaCy lemmatization call.

**Keep `Unit.speaker`.** It is dead today, but transcripts are the
project's stated primary input and a turn segmenter is a plausible near-term
addition; removing it would be churn. Instead, correct the `Unit` docstring
so it no longer implies a `"turn"` kind exists: state that `kind` is
currently `"block"` or `"sentence"`, and that `speaker` is reserved for a
future turn segmenter and is unpopulated by the built-in segmenters.

Update `tests/helpers.py` if `make_mention` passes the removed fields.

**Tests:**

- Update `test_types.py` to drop assertions on the removed fields.
- Full suite green. This task's real test is that nothing broke — grep for
  `.head` and `.lemma` across `src/` and `tests/` before declaring done.

---

## Task 10 — Pooled-metric resampling without duplication artifacts

**Files:** `src/lattice/harness/stats/resample.py`,
`src/lattice/harness/stats/intervals.py`,
`src/lattice/harness/stats/report.py`,
`tests/harness/stats/test_resample.py`,
`tests/harness/stats/test_intervals.py`,
new `tests/harness/stats/test_pooled_resample.py`

This is the subtlest task in the plan. Read it fully before starting.

**The defect.** B³ and ARI are *pairwise* (degree-2) functionals: a
mention's score depends on which other mentions share its cluster. The
document bootstrap draws with replacement, and
`ClusteringMetric._aggregate` re-keys each drawn document by its position
in the draw (`clustering.py:69`), so a document drawn twice contributes two
mentions with the *same* predicted concept id and the *same* gold cluster
id — guaranteed to agree in both partitions. Confirmed on a two-document
corpus containing one deliberate cross-document over-merge:

```
point estimate      b3-f1 0.6667  ari 0.0000
draw [A,A]          b3-f1 1.0000  ari 1.0000   <- the error vanishes
draw [A,B]          b3-f1 0.6667  ari 0.0000
draw [B,A]          b3-f1 0.6667  ari 0.0000
draw [B,B]          b3-f1 1.0000  ari 1.0000
bootstrap mean      b3-f1 0.8333  UPWARD BIAS +0.1667
```

The expected distinct-document fraction converges to 1 − 1/e ≈ 0.632 and
does **not** shrink with n, so this is an inconsistent estimator, not a
small-sample effect.

`edge-f1` has the mirror-image problem: `_aggregate` unions per-document
`frozenset`s (`edge_f1.py:52-54`), so multiplicity collapses entirely and
each replicate evaluates ~63.2% of the corpus. The visible damage is in a
shipped report — `reports/intervals/m4-food/interval-report.json` has
**every interval excluding its own point estimate** (f1 estimate 0.3233,
CI [0.2968, 0.3108]).

**The fix: m-out-of-n subsampling without replacement.** Drawing without
replacement removes duplication entirely, which kills both artifacts at the
source. The cost is that a size-m subsample has larger variance than the
full sample, so the interval must be rescaled by the convergence rate
√(m/n) (Politis, Romano & Wolf, *Subsampling*, 1999).

**Required changes:**

1. Add to `bootstrap()` in `resample.py` a keyword-only parameter
   `scheme: Literal["resample", "subsample"] = "resample"`.
   - `"resample"` is today's behaviour, unchanged. Do not alter it — the
     `f1-at-k` macro path is a correct bootstrap and must stay
     bit-identical.
   - `"subsample"` draws `m = max(2, round(n / 2))` documents **without
     replacement** from the pool each iteration, using
     `rng.sample(pool, m)`. `fixed_doc_ids` are still always included
     and are not part of the sampled pool, exactly as today.
2. Record the scheme and the realized `m` and `n` on the returned bundle
   so downstream interval construction can apply the correction. Extend
   whatever structure `bootstrap()` returns with `scheme`, `m`, and `n`
   fields; if it currently returns a bare list, introduce a small frozen
   dataclass and update call sites.
3. In `intervals.py`, add a subsampling interval constructor. For
   subsample draws, the standard construction centres on the point
   estimate and rescales the deviations:

   ```
   tau = sqrt(m / n)
   lo  = estimate + tau * (percentile(draws, alpha/2)     - estimate)
   hi  = estimate + tau * (percentile(draws, 1 - alpha/2) - estimate)
   ```

   Label the method `"subsample"`. This construction always brackets the
   estimate when the draws straddle it and shrinks correctly when they do
   not — which is the concrete defect in the M4 reports.
4. In `report.py`, select the scheme by metric kind: `kind == "macro"`
   keeps `"resample"`; `kind == "pooled"` uses `"subsample"`. Holistic
   metrics are unchanged in this task. Emit the scheme, `m`, and `n` into
   the report JSON so a reader can tell which construction produced an
   interval.
5. Do **not** change BCa's formulas. They are correctly transcribed
   (verified against Efron & Tibshirani including the acceleration sign
   convention). BCa applies to the `"resample"` scheme only; the
   subsample path uses the rescaled-percentile construction above.

**Tests (this is where the existing suite failed to catch the bug):**

- **The regression test the project lacked.** For each pooled metric
  (`clustering`, `edge-f1`), assert equivalence of `_aggregate` against
  direct recomputation on a **non-identity** document multiset — not just
  the full set. The existing tests only check the identity resample,
  which is the one case that cannot fail. For the subsample scheme the
  multiset is a strict subset, so equivalence must hold exactly.
- The two-document over-merge counterexample above: under
  `scheme="subsample"` with m=1... note m is floored at 2, so use a
  four-document construction with the same property and assert the
  bootstrap mean is not biased upward relative to the point estimate by
  more than a stated tolerance.
- `scheme="resample"` produces bit-identical output to the pre-change
  implementation for a fixed seed — pin this with a hardcoded expected
  list so the macro path is provably untouched.
- Subsample draws contain no duplicate document ids.
- The subsample interval brackets the point estimate when draws straddle
  it, using a hand-computed example with the arithmetic shown.
- `test_clustering_resample.py:47-59` currently asserts the buggy
  behaviour is correct ("Duplicating A must reweight the b3-precision
  mean toward A") — its fixture is a perfectly-clustered singleton, the
  one case where duplication is harmless. Rewrite this test to assert the
  degree-2 property instead: duplication of a document involved in a
  cross-document error must NOT improve the metric.

**Report in your report file:** the regenerated interval for
`reports/intervals/m4-food` under the new scheme, and whether it now
brackets its point estimate.

---

## Task 11 — Continuous integration, and track the evidence

**Files:** new `.github/workflows/ci.yml`, `.gitignore`,
`pyproject.toml`

The project has 460 tests, 135 commits, and no CI. Everything is on
`main`, no branches. Separately, `reports/` is gitignored, so all sixteen
report directories backing the README's claims are untracked — a clone
cannot verify any published number.

**Required changes:**

1. Add a GitHub Actions workflow at `.github/workflows/ci.yml`:
   - Triggers: `push` and `pull_request`.
   - Runs on `ubuntu-latest`, Python 3.12 and 3.13 in a matrix
     (`requires-python = ">=3.12"`).
   - Installs with `uv` (`astral-sh/setup-uv@v3`, then `uv sync`).
   - Steps: `uv run ruff check .`, then
     `uv run pytest -q -m "not ml"`.
   - The `ml` marker is deselected because those tests need
     sentence-transformers, spaCy models, and downloaded datasets. Note
     this in a comment in the workflow so the exclusion is deliberate and
     visible, not silent.
2. Un-ignore the report artifacts so claims are verifiable from a clone.
   In `.gitignore`, replace the bare `reports/` line with a rule that
   keeps the small JSON/markdown reports and ignores nothing else in
   there — the reports are text and small. Verify with
   `du -sh reports/` before committing; if the directory exceeds ~5 MB,
   keep only `reports/**/sweep-report.md`, `reports/**/sweep-report.json`
   and `reports/intervals/**/*.json`, and say so in the commit message.
3. `git add` the report files and commit them as part of this task.

**Tests:** none beyond the suite staying green. Validate the workflow YAML
parses (`python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml'))"`).

---

## Task 12 — Regenerate the affected benchmark reports

**Files:** `reports/**` (regenerated), `docs/results/2026-07-31-post-fix.md` (new)

Tasks 4, 5, 7, 8 and 10 change measured numbers. Nothing may be quoted
until it is regenerated (GC8).

**Required runs** — each writes to its existing report directory via
`PYTHONPATH=src .venv/bin/python -m lattice.harness --sweep <config> <out_dir>`:

| config | out_dir | why |
|---|---|---|
| `configs/m2b-sweep.toml` | `reports/m2b` | HCUKE fix (T4) + F1@K denominator (T5) |
| `configs/m2a-baseline-sweep.toml` | `reports` | F1@K denominator (T5) |
| `configs/m3-conel2-stemmed.toml` | `reports/m3-conel2-stemmed` | new three-way identity baseline (T7) |
| `configs/m3-ecbplus-stemmed.toml` | `reports/m3-ecbplus-stemmed` | new three-way identity baseline (T7) |
| `configs/m5-conel2-nn090.toml` | `reports/m5-conel2` | decoupled redundancy threshold (T8) |

These runs need MiniLM (cached under the HF cache) and the datasets under
`data/`, both already present. Expect the M2b sweep to take the longest —
MDERank and HCUKE issue many embedder calls. If a run exceeds 30 minutes,
report it rather than killing it.

Then write `docs/results/2026-07-31-post-fix.md` recording, for each track:
the before number, the after number, the config that produced it, and a
one-line reading. Follow the house style of `docs/results/2026-07-25-m5-cross-corpus.md`.
State plainly where a fix made lattice's own default look *less*
favourable — that is the point of the exercise, and the project has a
track record of doing exactly this (commit `f34205d`).

**Do not** re-run the M4 TExEval sweeps or the interval analysis in this
task; T10's report covers the interval change and no T1–T11 change affects
M4 edge extraction except T2, which is covered in T13.

---

## Task 13 — Re-run M4 with the Hearst coordination fix

**Files:** `reports/m4-*` (regenerated), `docs/results/2026-07-31-post-fix.md` (extended)

Task 2 increases Hearst recall on backward-coordinated patterns, which
changes the M4 hierarchy numbers — the one track whose published result
currently holds up. Re-run all six golds:

```
configs/m4-food-sweep.toml            -> reports/m4-food
configs/m4-food-wordnet-sweep.toml    -> reports/m4-food-wordnet
configs/m4-science-sweep.toml         -> reports/m4-science
configs/m4-science-wordnet-sweep.toml -> reports/m4-science-wordnet
configs/m4-science-eurovoc-sweep.toml -> reports/m4-science-eurovoc
configs/m4-env-eurovoc-sweep.toml     -> reports/m4-env-eurovoc
```

Append an M4 section to `docs/results/2026-07-31-post-fix.md` with the
before/after F1 per gold, and re-check the published-band claim: the
design spec's participant ranges are at
`docs/2026-07-12-m4-hierarchy-design.md:299-305`. Report how many golds
now sit above the top of their band.

---

## Task 14 — Correct the README

**Files:** `README.md`

Do this last; it depends on T12 and T13 output. Every number must come
from a regenerated report (GC8).

The README was committed once (`02d5dc4`) and never revised, and the
credibility work that landed afterward repudiates parts of it.

**Required corrections:**

1. **Identity row — the most serious error.** The README quotes
   `0.643` and `0.962`. Those are the **argmax over the threshold grid**,
   at 0.75 for ECB+ and 0.80 for ConEL-2 — two different thresholds,
   *neither of which is the shipped 0.90*. The shipped configuration
   scores **0.6255** and **0.9491**. The project's own spec names this
   trap explicitly at
   `docs/2026-07-14-statistical-intervals-design.md:183-186`:
   "Comparing exact-label against the maximum over thresholds (0.9620,
   which sits at 0.80, not the operating point) would be selection bias."
   Replace with the shipped numbers and their paired CIs from
   `reports/intervals/analysis/m3-paired-delta.json`
   (ConEL-2 Δ=+0.0105 [0.0029, 0.0163]; ECB+ Δ=+0.0172 [0.0096, 0.0216]),
   and add the new `stemmed-label` row from T12 so the reader can see what
   the embedding contributes over stemming.
2. **Salience row.** Label `frequency` as what its own docstring calls it —
   a trivial walking-skeleton baseline, not a competitive method. Use the
   regenerated T12 numbers. State that no classical unsupervised baseline
   (TF-IDF, TextRank, YAKE) is implemented, so the comparison is
   in-project only.
3. **Hierarchy row.** Add the caveat that TExEval-2 ships no corpus and
   lattice supplies Wikipedia summaries — definitional, copula-rich text
   that is materially easier than what the 2016 participants used. The
   band comparison is therefore not like-for-like. Use T13's numbers.
4. **Integration row.** The `duplicate-rate 0.117 → 0.015` claim is a
   theorem under the old config (metric threshold == resolver threshold).
   Replace with the T12 rerun at the decoupled threshold.
5. **State n beside every headline**: Inspec 500 documents, ConEL-2 58
   conversations / 452 mentions, ECB+ 206 documents / 2054 mentions.
6. **The "every default was chosen by benchmark" claim** (line 12) is not
   true of the `lite` profile: it ships `embedding-nn @ 0.90`, a threshold
   calibrated entirely in MiniLM cosine space, with the **hashing trigram**
   embedder, where the similarity geometry is unrelated. Either qualify
   the claim to the `standard` profile or drop the evidence column for
   `lite`.
7. **The opening pitch.** The README leads with "recognizing that 'vector
   store' in session 5 and 'vector database' in session 200 are the same
   concept". Verified on the shipped `standard` profile: those two remain
   **separate concepts** (cosine 0.681 < threshold 0.90). Rewrite the
   pitch so it describes what the engine does today — cross-document
   identity for repeated and morphologically-varied surface forms — and,
   if you want to keep the synonym example, mark it explicitly as the
   roadmap target it is. Do not delete the ambition; state it honestly.

**Tests:** `tests/api/test_readme.py` extracts and executes the first
fenced python block — keep it passing. If you change the quickstart code,
run that test specifically.

---

## Out of scope (recorded, not done here)

- **Contextual resolution.** `Mention.context` is populated by all four
  extractors and read by nothing; embedding the mention in context rather
  than the bare surface is the real fix for the synonym-merging failure.
  It is an algorithm change needing its own design spec and sweep.
- **numpy-backed concept store.** `InMemoryConceptStore.nearest` is a
  brute-force scan with a pure-Python cosine: 27.7 µs per comparison,
  275 ms/query at 10k concepts, 689 ms at 25k; `Engine.ingest` reaches
  125 ms/doc at 2,500 concepts. The `ConceptStore` port makes this a
  drop-in replacement, but numpy is not a core dependency (GC9), so it
  belongs behind the `ml` extra as a separate adapter.
- **Concept aliases.** The merge decision is the engine's core artifact
  and is never persisted — `Concept` has no alias field and `save()`
  writes none. Needs a `format_version` bump.
- **Holistic-metric resampling.** T10 fixes the pooled path only. The
  holistic path re-runs the pipeline on a document multiset, and
  duplicated documents re-resolve into their own concepts
  (`coherence.singleton-fraction` measured 0.5 → 0.0 under duplication).
- **Multiplicity correction.** ~53 sweep rows and ~70 published intervals
  with no family-wise error control.
