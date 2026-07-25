# Interval & Permutation Analysis — M2 through M5

This is the Task 11 deliverable of the `lattice.harness.stats` credibility
track: it runs the interval/permutation layer (Tasks 1–10) on the real
corpora (ConEL-2, ECB+, Inspec, TExEval-2) with the real ML models (spaCy
`en_core_web_sm`, sentence-transformers `all-MiniLM-L6-v2`), and reports every
number alongside the exact command that regenerates it. No number in this
document is estimated, interpolated, or fabricated — each was produced by
running the recorded command against this repository's checked-in data and
code, on 2026-07-18.

## Methodology recap

- **Item-level bootstrap** (M2, M3, M4): one real pipeline run captures a
  per-document `ResampleBundle`; `bootstrap()` resamples document ids with
  replacement and recomputes the metric's own `aggregate()` B=10000 times
  (the CLI's default); `jackknife()` performs one leave-one-out pass for the
  BCa acceleration term. Cheap — no pipeline re-run per resample.
- **Holistic bootstrap** (M5): `bootstrap_holistic()` re-runs the *entire*
  pipeline (segmenter → extractor → scorer → resolver → relation inducer →
  graph integrator → metric) per resample, B=1000 (the CLI's `--holistic`
  default), because M5's metrics (`redundancy`, `hierarchy-sanity`,
  `coherence`) are `kind = "holistic"` — they judge the accreted graph as a
  whole, not per-document records.
- **Intervals**: BCa (bias-corrected, accelerated) is reported where it is
  well-defined; the plain percentile interval is always available alongside.
  `bca_interval` falls back to the percentile interval (labelled
  `"percentile-fallback"`) in two degenerate regimes: (a) every resample lies on
  one side of the point estimate (`prop ∈ {0,1}`, bias term undefined), and (b)
  the computed BCa interval fails to bracket the point estimate (extreme
  bias/acceleration collapsed it — see the §2.2 skewed-resample caveat). Both
  guards are unit-tested (`tests/harness/stats/test_intervals.py`). Guard (b) was
  added after the whole-branch review surfaced collapsed BCa intervals on skewed
  cells; the affected M3/M4 reports were regenerated with it (2026-07-24).
- **Paired delta**: `bootstrap()` run with the *same seed* on two configs'
  clustering bundles draws identical document indices at each iteration i
  (paired by construction, since both bundles share the same `per_document`
  key ordering from the same dataset), then `paired_delta()` computes the
  resample-wise difference, its percentile CI, and `prob_positive` (the
  fraction of paired resamples where the delta is positive).
- **Order-permutation**: `order_spread()` runs the pipeline over K=40 seeded
  shuffles of the document stream (holding any `fixed_prefix` documents
  first) and reports the range/std of each metric across the K orderings —
  a robustness check, not a hypothesis test.
- All runs use `--seed 0` (bootstrap/jackknife) unless noted; order-permutation
  uses `seed=1` (`scripts/interval_analysis.py`'s `PERMUTATION_SEED`).

## 1. Headline: does nn@0.90 beat exact-label on b3-f1?

**Yes, on both corpora — the paired-bootstrap delta excludes zero, though the
absolute margin is small.**

- **ConEL-2** (58 docs): nn@0.90 b3-f1 0.9491 − exact-label 0.9386 = **Δ +0.0105**,
  95% CI **[0.0029, 0.0163]**, P(Δ>0) = 0.999 (B=10000, seed 0).
- **ECB+** (206 docs): nn@0.90 b3-f1 0.6255 − exact-label 0.6084 = **Δ +0.0171**,
  95% CI **[0.0096, 0.0216]**, P(Δ>0) = 1.000 (B=10000, seed 0).

The result **replicates across two independent corpora**, with the operating point
(nn@0.90) fixed on ConEL-2 in M5 and applied out-of-sample to ECB+ (not re-tuned).

**Why the paired test matters — and why the marginal CIs look like they disagree.**
The per-config marginal b3-f1 CIs (§2.2) overlap heavily: ConEL-2 nn@0.90 is
[0.9267, 0.9552] and exact-label is [0.9204, 0.9442]. Read alone, those overlapping
intervals would suggest "no significant difference." That reading is wrong: the two
resolvers are scored on the *same* documents, so most of the b3-f1 variance is shared
corpus variance that cancels in the difference. The paired bootstrap (identical
resampled document indices for both configs at each iteration) measures the *delta's*
variance directly, and that CI excludes zero. This is exactly why the spec
pre-registered a paired delta instead of comparing two marginal CIs — a ~0.01
point-estimate gap that looks like noise beside the marginal CIs is a real, consistent
within-resample improvement.

**Honest caveat on effect size:** "beats" is defensible, but the effect is small
(~0.011 b3-f1 on ConEL-2, ~0.017 on ECB+). embedding-nn@0.90 is a modest,
statistically-detectable improvement over exact-label on clustering quality — not a
large one. On ECB+ the *marginal* intervals carry the boundary caveat in §2.2 (the
point estimate can fall outside the resample range); the *paired* delta is computed on
the resample-wise difference and is not affected by that pathology (P(Δ>0)=1.0 means
all 10000 paired resamples favored nn@0.90).

## 2. Per-milestone CI tables

### 2.1 M2 — Inspec f1@k (embedding-cosine scorer, top_k=15)

Config: `configs/m2b-f1atk.toml`, collapsed from `configs/m2b-sweep.toml`'s
`[base.*]`. The M2b sweep varies 4 scorers (frequency/embedding-cosine/mderank/hcuke);
`embedding-cosine` was picked as the single representative because it is the
sweep's own `[base.scorer]` (the axis exists to swap it out, so the base value
is the sweep author's natural default) and it needs no extra dependencies
beyond the shared sentence-transformer embedder already required elsewhere in
this doc. `top_k=15` matches the sweep's own axis params (`f1-at-k` requires
`scorer.top_k >= max(ks) = 15`).

Command:
```
python -m lattice.harness.stats configs/m2b-f1atk.toml reports/intervals/m2b --seed 0
```
(B=10000, seed=0, 500-document Inspec test split.)

| metric | estimate | BCa 95% CI |
|---|---|---|
| precision@5  | 0.4144 | [0.3938, 0.4351] |
| recall@5     | 0.2545 | [0.2380, 0.2720] |
| f1@5         | 0.2967 | [0.2804, 0.3135] |
| precision@10 | 0.3551 | [0.3392, 0.3709] |
| recall@10    | 0.4004 | [0.3811, 0.4212] |
| f1@10        | 0.3547 | [0.3401, 0.3695] |
| precision@15 | 0.3068 | [0.2929, 0.3205] |
| recall@15    | 0.4796 | [0.4594, 0.5009] |
| f1@15        | 0.3555 | [0.3417, 0.3694] |

### 2.2 M3 — clustering b3 metrics, both corpora, both configs

Configs: `configs/m3-{conel2,ecbplus}-{exact,nn090}.toml`, collapsed from
`configs/m3-{conel2,ecbplus}-sweep.toml`'s `[base.*]` + the `exact-label` /
`embedding-nn threshold=0.90` axis options. **The nn@0.90 operating point was
chosen on ConEL-2 in M5 and is applied uniformly to ECB+ — not re-tuned per
corpus** (task-11-brief-v2 scope amendment). ConEL-2 test split: 58 documents.
ECB+ test split: 206 documents.

Commands:
```
python -m lattice.harness.stats configs/m3-conel2-exact.toml   reports/intervals/m3-conel2-exact   --seed 0
python -m lattice.harness.stats configs/m3-conel2-nn090.toml   reports/intervals/m3-conel2-nn090   --seed 0
python -m lattice.harness.stats configs/m3-ecbplus-exact.toml  reports/intervals/m3-ecbplus-exact  --seed 0
python -m lattice.harness.stats configs/m3-ecbplus-nn090.toml  reports/intervals/m3-ecbplus-nn090  --seed 0
```
(B=10000, seed=0 each.)

**ConEL-2:**

| config | b3-precision | b3-recall | b3-f1 | ari |
|---|---|---|---|---|
| exact-label | 0.9978 [0.9918, 1.0000] | 0.8861 [0.8538, 0.8971] | 0.9386 [0.9204, 0.9442] | 0.7605 [0.8022, 0.9303]† |
| nn@0.90     | 0.9978 [0.9918, 1.0000] | 0.9050 [0.8636, 0.9167] | 0.9491 [0.9267, 0.9552] | 0.8036 [0.8278, 0.9468]† |

**ECB+:**

| config | b3-precision | b3-recall | b3-f1 | ari |
|---|---|---|---|---|
| exact-label | 0.8077 [0.8238, 0.8593]† | 0.4880 [0.5034, 0.5462]† | 0.6084 [0.6293, 0.6636]† | 0.3282 [0.3456, 0.4138]† |
| nn@0.90     | 0.7842 [0.8028, 0.8399]† | 0.5203 [0.5339, 0.5751]† | 0.6255 [0.6458, 0.6782]† | 0.3780 [0.3903, 0.4563]† |

† **Skewed-resample caveat (a real property, handled uniformly).** The cells marked †
report the **percentile** interval, not BCa. On these cells (nearly) every one of the
10000 resamples falls on the *same side* of the full-corpus point estimate — the plug-in
estimate sits outside the range its own resamples span. This is a real property of the
mention-weighted, multiplicity-preserving pooled statistic (clustering's `_aggregate`
re-indexes duplicated documents rather than collapsing them — spec-intended, Execution
Amendment #4): resampling documents with replacement shifts the relative weight of
high-mention vs. low-mention documents, and B³'s ratio structure means that shift does
not average back to the plug-in value. It affects **all ECB+ clustering cells** *and* the
**ConEL-2 `ari`** cells above (and M4 `food`/`food-wordnet` precision, §2.3) — so it is a
property of skewed corpora, **not ECB+-specific**. In this regime BCa's bias/acceleration
adjustment is undefined or degenerate — it can collapse to a zero-width or non-bracketing
interval — so `bca_interval` detects the two degenerate cases (`prop ∈ {0,1}`, or a
computed interval that fails to bracket the estimate) and falls back to the plain
percentile interval, labelled `"percentile-fallback"` in the JSON. Read these bounds as
*where the resamples concentrate*; they may not bracket the point estimate, so do not
over-index on their exact values. **The §1 b3-f1 headline is unaffected** — the paired
delta is computed on the resample-wise difference, not on these marginal intervals.

### 2.3 M4 — edge-f1 P/R/F1, per gold (union = hearst + compound)

Configs: `configs/m4-{gold}-union.toml`, collapsed from
`configs/m4-{gold}-sweep.toml`'s `[base.*]` + the `union` axis option.
Glossary-first document ordering (`src/lattice/adapters/dataset/taxonomy.py`):
the glossary document (stream position 0) is held fixed via `--fixed-prefix 1`.

Commands:
```
for g in env-eurovoc food food-wordnet science science-eurovoc science-wordnet; do
  python -m lattice.harness.stats configs/m4-$g-union.toml reports/intervals/m4-$g --seed 0 --fixed-prefix 1
done
```
(B=10000, seed=0 each.)

| gold | docs | gold edges | precision | recall | f1 |
|---|---|---|---|---|---|
| env-eurovoc     |  180 |  261 | 0.6375 [0.6329, 0.6456] | 0.1954 [0.1916, 0.1954] | 0.2991 [0.2941, 0.3000] |
| food            | 1312 | 1587 | 0.6148 [0.6154, 0.6397]† | 0.2193 [0.1953, 0.2060] | 0.3233 [0.2968, 0.3108] |
| food-wordnet    | 1117 | 1533 | 0.6234 [0.6258, 0.6449]† | 0.2818 [0.2603, 0.2707] | 0.3881 [0.3682, 0.3807] |
| science         |  302 |  465 | 0.6839 [0.6765, 0.6882] | 0.2559 [0.2516, 0.2559] | 0.3725 [0.3673, 0.3748] |
| science-eurovoc |  120 |  124 | 0.6000 [0.5862, 0.6000] | 0.1452 [0.1371, 0.1452] | 0.2338 [0.2222, 0.2338] |
| science-wordnet |  355 |  441 | 0.8056 [0.8019, 0.8113] | 0.1973 [0.1927, 0.1973] | 0.3169 [0.3119, 0.3181] |

(Sanity check: F1 is the harmonic mean of precision and recall, so it must
lie between them — verified for every row above, e.g. science-wordnet:
`2·0.8056·0.1973/(0.8056+0.1973) = 0.3170 ≈ 0.3169`. science-wordnet is a
high-precision, low-recall gold relative to its siblings.)

**Edge-set-pooled bootstrap semantics (applies to all of M4):** `EdgeF1`'s
`kind = "pooled"` bundle stores each document's IS_A edges as a `frozenset`;
`_aggregate` takes `predicted |= record` — a **set union** across the
resampled documents. Unlike clustering's index-prefixing (§2.2), duplicate
documents in a bootstrap resample contribute **no additional distinct
edges** — multiplicity collapses by construction. This means the M4 bootstrap
CI does **not** reflect "what if we saw this document again" variance; it
reflects **corpus-composition variance** — which subset of the real,
finite edge set would survive if the document sample had been different. Two
consequences are visible above. First, the glossary document (held fixed via
`--fixed-prefix 1`) contributes the compound edges to *every* resample, so those
edges never drop out; the CIs are correspondingly tight and reflect only the
corpus-document (Hearst) edge composition. Second, the intervals are
**one-sided**: because set union can only *lose* edges when documents are
dropped, no resample exceeds the full-corpus coverage, so the point estimate
sits at or above the *upper* edge of its recall/F1 CI — e.g. env-eurovoc recall
0.1954 with CI `[0.1916, 0.1954]` (estimate exactly at the top), or food recall
0.2193 with CI `[0.1953, 0.2060]` (estimate *above* the top, because
with-replacement resampling drops ~37% of documents on average and their Hearst
edges with them). This one-sidedness is intrinsic to edge-set pooling, not an
artifact — it is the honest shape of corpus-composition uncertainty for a
union-pooled metric. On the two largest golds it becomes extreme enough that the
`food`/`food-wordnet` **precision** cells (marked †) have the point estimate fall
just below their resample range, so BCa degenerates and they report the percentile
interval — the same skewed-resample regime documented in the §2.2 caveat.

## 3. M3 threshold-sensitivity curve

`scripts/interval_analysis.py`, grid 0.65–0.95 step 0.05, B=10000, seed 0. The
nn@0.90 operating point (chosen in M5 for the redundancy↔coherence tradeoff, not for
b3-f1) is marked.

**conel2:**

| threshold | b3-f1 | 95% CI | method |
|---|---|---|---|
| 0.65 | 0.9091 | [0.8666, 0.9188] | bca |
| 0.70 | 0.9304 | [0.8892, 0.9420] | bca |
| 0.75 | 0.9457 | [0.9012, 0.9573] | bca |
| 0.80 | 0.9620 | [0.9330, 0.9710] | bca |
| 0.85 | 0.9562 | [0.9407, 0.9627] | bca |
| **0.90** ← op | 0.9491 | [0.9267, 0.9552] | bca |
| 0.95 | 0.9386 | [0.9204, 0.9442] | bca |

On ConEL-2, b3-f1 peaks at threshold **0.80** (0.9620); the operating point 0.90
(0.9491) sits just past the peak — confirming 0.90 is *not* b3-f1-optimal (it was
chosen for M5's intrinsic tradeoff). But the per-threshold CIs overlap heavily across
0.75–0.90, so the curve is a broad plateau rather than a sharp peak: the operating
point within that band is not sharply distinguished by b3-f1. At 0.95, b3-f1 equals the
exact-label value (0.9386) — as expected, a high threshold merges almost nothing and
approaches exact-label behavior.

**ecbplus:**

| threshold | b3-f1 | 95% CI | method |
|---|---|---|---|
| 0.65 | 0.6240 | [0.6433, 0.6740] | percentile-fallback |
| 0.70 | 0.6411 | [0.6599, 0.6897] | percentile-fallback |
| 0.75 | 0.6428 | [0.6616, 0.6921] | percentile-fallback |
| 0.80 | 0.6421 | [0.6611, 0.6915] | percentile-fallback |
| 0.85 | 0.6404 | [0.6597, 0.6907] | percentile-fallback |
| **0.90** ← op | 0.6255 | [0.6458, 0.6782] | percentile-fallback |
| 0.95 | 0.6188 | [0.6394, 0.6734] | percentile-fallback |

ECB+ carries the §2.2 boundary caveat throughout (every row is `percentile-fallback`,
and the point estimate can sit outside its own CI); read these as directional. The
plateau is even flatter (0.70–0.85 all ≈ 0.64).

## 4. M5 — intrinsic metrics, holistic bootstrap

`configs/m5-conel2-nn090.toml`, `--holistic`. **B=150** — reduced from the CLI's 1000
default: each resample re-runs the entire pipeline (~20s/resample), so B=1000 would be
~5.5 hours; B=150 keeps the run to ~50 min while still giving a serviceable percentile
CI. This reduction is recorded here, not silently defaulted. seed 0; percentile
intervals (holistic BCa acceleration would need a pipeline jackknife, not computed).

| metric | key | estimate | 95% CI (percentile) |
|---|---|---|---|
| coherence | coherence | 0.9311 | [0.9282, 0.9404] |
| coherence | multi-surface-concepts | 27.0 | [11.0, 20.3] |
| coherence | singleton-fraction | 0.7954 | [0.3571, 0.5623] |
| hierarchy-sanity | is-a-edges | 88.0 | [42.0, 68.0] |
| hierarchy-sanity | max-depth | 2.0 | [2.0, 3.0] |
| hierarchy-sanity | cycle-components | 0.0 | [0.0, 0.0] |
| hierarchy-sanity | cycle-nodes | 0.0 | [0.0, 0.0] |
| hierarchy-sanity | self-loops | 0.0 | [0.0, 0.0] |
| hierarchy-sanity | transitive-shortcuts | 0.0 | [0.0, 0.0] |
| redundancy | concept-count | 523.0 | [303.2, 376.6] |
| redundancy | duplicate-rate | 0.0153 | [0.0000, 0.0244] |
| redundancy | near-duplicate-pairs | 4.0 | [0.0, 4.0] |

The intrinsic families behave as designed: `hierarchy-sanity` shows zero
cycles/self-loops/shortcuts and shallow depth (the induced IS_A graph is a clean,
mostly-flat forest); `coherence` is high (0.93 — merged surfaces are semantically
tight); `redundancy` duplicate-rate is low (0.015). The count-like keys (concept-count,
is-a-edges) have CIs that sit *below* their full-corpus point estimate — a direct
consequence of holistic resampling: dropping documents removes whole swaths of the
accreted graph, so every resample has fewer concepts/edges than the full run, and the
point estimate sits at the high end.

## 5. Order-permutation stability (K=40)

`scripts/interval_analysis.py`, K=40 seeded shuffles (seed 1), reporting the
range/std of each metric across the orderings.

- **M4 (all six golds, `fixed_prefix=1`):** range **exactly 0** on every metric of
  every gold. The glossary-first pipeline is perfectly order-invariant — corpus
  document order cannot change the induced edge set; only the glossary (held fixed at
  stream position 0) seeds terms. This validates the M4 design claim directly.
- **M3 ConEL-2 (nn@0.90):** range ≤ 8e-16 (floating-point noise) — order-invariant.
- **M3 ECB+ (nn@0.90):** small but non-zero — b3-f1 range 0.0037, b3-recall 0.0055,
  ari 0.0066. ECB+'s larger, more entangled cross-document clusters give
  embedding-nn's greedy arrival-order merging a slight foothold; the effect is
  negligible (< 0.007).
- **M5 ConEL-2 (nn@0.90):** small — concept-count range 2.0, is-a-edges range 7.0,
  coherence range 0.0026; cycles/self-loops/shortcuts invariant at 0.

Order effects are negligible everywhere and exactly zero where the design guarantees
it (M4). The permutation harness itself is mechanism-tested (Task 9) to confirm it
genuinely varies document order rather than silently no-opping.

## 6. Regenerating every number

```bash
# environment (see memory: macOS venv .pth hidden-flag quirk)
chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null
export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())")

# M3 item-level (B=10000), both corpora, both configs
uv run --no-sync python -m lattice.harness.stats configs/m3-conel2-exact.toml   reports/intervals/m3-conel2-exact   --seed 0
uv run --no-sync python -m lattice.harness.stats configs/m3-conel2-nn090.toml   reports/intervals/m3-conel2-nn090   --seed 0
uv run --no-sync python -m lattice.harness.stats configs/m3-ecbplus-exact.toml  reports/intervals/m3-ecbplus-exact  --seed 0
uv run --no-sync python -m lattice.harness.stats configs/m3-ecbplus-nn090.toml  reports/intervals/m3-ecbplus-nn090  --seed 0

# M2 Inspec f1@k
uv run --no-sync python -m lattice.harness.stats configs/m2b-f1atk.toml reports/intervals/m2b --seed 0

# M4, one per gold, glossary held fixed
for g in env-eurovoc food food-wordnet science science-eurovoc science-wordnet; do
  uv run --no-sync python -m lattice.harness.stats configs/m4-$g-union.toml reports/intervals/m4-$g --seed 0 --fixed-prefix 1
done

# M5 intrinsic, holistic (full pipeline re-run per resample — slow; B=150 keeps it ~50 min)
uv run --no-sync python -m lattice.harness.stats configs/m5-conel2-nn090.toml reports/intervals/m5-conel2 --holistic --samples 150 --seed 0

# M3 paired delta + threshold curve (both corpora) + order-permutation (M3 both corpora, M4 six golds, M5)
uv run --no-sync python scripts/interval_analysis.py reports/intervals/analysis
```
