# lattice — Statistical Intervals Design Spec

**Date:** 2026-07-14
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` (milestone list complete;
this is additive credibility hardening, not a parent-spec milestone)
**Status:** approved design, pre-plan
**Track:** 1 of 3 credibility-hardening sub-projects (this; then a second M5 corpus;
then an excluded-method baseline — each its own spec → plan → execution cycle)

Every headline number in the M2–M5 reports is a single point estimate on one fixed
split. This sub-project turns those point estimates into interval claims: bootstrap
confidence intervals, a paired-delta test for the load-bearing comparative claim
("embedding-nn beats exact-label"), threshold-sensitivity curves that show the chosen
operating point is not a fragile peak, and order-permutation robustness reports. The
machinery ships as a first-class, tested part of the harness — reproducible, seeded,
regenerable — not a one-off analysis pass.

## 1. Goal

Deliver a shipped `lattice.harness.stats` layer that, driven by the existing sweep
configs and touching no existing behavior, produces for M2/M3/M4/M5: (a) BCa 95%
confidence intervals on the reported metrics, (b) a paired-bootstrap CI on the
M3 resolver delta, (c) threshold-sensitivity curves with per-point CI bands, and
(d) order-permutation spread reports. Item-level metrics bootstrap from a **single**
pipeline run; the holistic M5 metrics re-run the pipeline per resample (tractable only
because ConEL-2 is 58 conversations). Output is a regenerable JSON report plus one
committed, citable results document.

## 2. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Scope breadth | Full matrix — all three techniques across M2/M3/M4/M5 wherever each is meaningful | User decision 2026-07-14. Credibility hardening should cover every published number, not just the one comparative claim. |
| Durability | Shipped, unit-tested harness capability (`src/lattice/harness/stats/`), regenerable via CLI | User decision 2026-07-14. A tested statistical layer is more defensible than a script whose numbers can drift from the code; matches how M2–M6 were built. |
| Detail wiring | Resampling-primitive protocol: each metric declares its kind and emits per-document records; the stats engine carries one recompute kernel per kind | User decision 2026-07-14. Maps onto the three statistical shapes we actually have; keeps `Metric.evaluate` frozen (additive). |
| Resampling unit | The document (conversation) — uniformly, across all kinds | Documents are the independent sampling units; corpus-composition variance is the honest thing to quantify. Edge/mention attribution rides on existing structure (`Relation.provenance`, `GraphDelta.resolutions`). |
| CI method | BCa 95% primary; percentile reported alongside | BCa corrects bias and skew (publication-grade); the leave-one-document-out jackknife it needs is cheap item-level and affordable holistic (58 conversations). |
| "Seeds" explicitly rejected | Do not vary training seeds | The pipeline is deterministic (no stochastic training/sampling); multiple seeds return identical numbers. The variance that matters is over the evaluation set (bootstrap) and document ordering (permutation). Documented, not silently omitted. |

## 3. Core / protocol changes (all additive)

No existing type, port method, adapter behavior, config, or recorded result changes.
`Metric.evaluate` / `DocumentMetric.evaluate_documents` stay exactly as they are.

A new opt-in mixin makes a metric resamplable:

```python
class Resamplable:
    kind: str  # "macro" | "pooled" | "holistic"

    def emit_records(self, evaluation_input, ground_truth) -> ResampleBundle: ...
    # holistic metrics inherit a default that raises / signals "re-run me"
```

- `ResampleBundle` (a frozen dataclass in the stats package) carries, per document id,
  that document's contribution, plus any global context the kernel needs:
  - **macro** (f1@k): `per_document[doc_id] = {"f1@5": ..., "precision@5": ..., ...}` —
    the document's own metric values. Aggregate = mean over sampled documents.
  - **pooled** (edge-F1, B³/ARI): `per_document[doc_id] = <the document's items>` plus
    `global_context = <fixed gold set / clustering>`. Aggregate = pool sampled
    documents' items, recompute the ratio/clustering formula.
  - **holistic** (M5 intrinsic): emits nothing; `kind="holistic"` signals the engine to
    re-run the pipeline on the resampled document multiset.
- **Equivalence requirement (test-enforced, §7):** for every resamplable metric,
  `aggregate(emit_records(...))` over the full document set must equal the metric's own
  `evaluate(...)` output exactly. The point estimate and the interval's center share one
  code path — no drift.

The metrics that opt in: `f1-at-k` (macro), `edge-f1` (pooled), the M3 clustering
metric `clustering` (pooled), and the three M5 intrinsic metrics `redundancy`,
`coherence`, `hierarchy-sanity` (holistic). `label-f1` and `co-occurrence`-only configs
are not opted in (not part of any headline claim); the engine reports them as
"not resamplable" rather than failing.

## 4. New modules — `src/lattice/harness/stats/`

### 4.1 `resample.py` — the engine and the three kernels

```python
def bootstrap(bundle: ResampleBundle, kind: str, *, samples: int, seed: int
              ) -> list[dict[str, float]]
def bootstrap_holistic(config: ExperimentConfig, metric_specs, *, samples: int,
                       seed: int) -> list[dict[str, float]]
```

- One seeded `random.Random(seed)`. Each of the `samples` iterations draws
  `len(documents)` document ids with replacement and recomputes via the kind's kernel,
  yielding one dict of metric values per iteration (the resample distribution).
- **macro kernel:** mean of the sampled documents' per-doc value dicts (multiplicity
  respected — a document drawn twice counts twice).
- **pooled kernel:** union/concatenate the sampled documents' items, recompute the
  formula. Mention-pooled metrics (B³/ARI) are multisets — multiplicity is respected and
  the bootstrap is standard. Edge-set-pooled (edge-F1) dedups predicted edges into a set,
  so multiplicity collapses **by construction**: the resulting CI reflects
  *which corpus documents are present*, the intended corpus-composition variance. This
  semantics is stated explicitly in the results doc, not hidden.
- **holistic kernel** (`bootstrap_holistic`): for each iteration, materialize the
  resampled document multiset, run `run_experiment` on it, read the metric off the
  fresh report. `samples` is smaller here (§5).
- **M4 glossary handling:** the glossary document (doc 0, the term inventory) is held
  fixed in every resample; only the corpus documents (Wikipedia summaries) are resampled.
  Compound edges evidenced by the glossary are corpus-invariant and stay in every sample.

### 4.2 `intervals.py` — CI construction

```python
def bca_interval(estimate: float, resamples: list[float], jackknife: list[float],
                 *, level: float = 0.95) -> Interval          # (lo, hi, method)
def percentile_interval(estimate: float, resamples: list[float],
                        *, level: float = 0.95) -> Interval
def paired_delta(resamples_a: list[float], resamples_b: list[float]
                 ) -> DeltaResult   # estimate, (lo, hi), prob_positive
```

- BCa: bias-correction `z0` from the fraction of resamples below the point estimate;
  acceleration `a` from leave-one-document-out jackknife values. Percentile is always
  computed too, so the doc can show both.
- **Degenerate guards:** if all resamples are identical (zero variance — e.g. an
  order-invariant sanity check), return a zero-width interval at the estimate rather than
  dividing by zero; if `z0` is undefined (all resamples on one side), fall back to
  percentile and flag it in the output.
- `paired_delta`: consumes two resample lists produced from the **same** per-iteration
  document draws, returns the delta point estimate, its 95% CI, and
  `prob_positive = mean(delta_i > 0)`. Mechanism: both configs' `bootstrap()` are run
  with the **same seed** over the same document set, so iteration *i* draws identical
  document indices for both — the two lists are paired element-wise with no dedicated
  paired engine. (Valid because M3 configs differ only in the resolver; both process the
  same corpus with the same document ids.)

### 4.3 `permutation.py` — order-stability

```python
def order_spread(config: ExperimentConfig, metric_specs, *, permutations: int,
                 seed: int) -> SpreadResult   # values, min, max, range, std
```

- K seeded random orderings of the document stream; run the pipeline on each; report the
  spread of each metric. A robustness report, not a hypothesis test. Applies where the
  graph accretes (M3, M5). For M4 it is a **sanity check**: the glossary-first protocol
  is order-independent by design, so near-zero spread is the expected, design-validating
  result and is reported as such.

### 4.4 `runner.py` addition — `run_experiment_detailed`

A sibling of `run_experiment` that runs the pipeline once and returns the `RunReport`
plus a `dict[str, ResampleBundle]` (one bundle per resamplable metric), captured in the
same pass. Item-level bootstrap consumes this without re-running. `run_experiment` is
untouched.

### 4.5 `__main__.py` — CLI

`python -m lattice.harness.stats <config.toml> <out_dir> [flags]`

- `--samples N` (default 10000 item-level, 1000 holistic — the module picks by kind
  unless overridden), `--seed S` (default fixed), `--thresholds lo:hi:step` (enables the
  sensitivity curve over the resolver axis), `--permutations K` (default 0 = off; 40
  when requested), `--paired A,B` (name two resolver configs for the delta).
- Reuses the existing sweep TOMLs. Writes `interval-report.json` to `<out_dir>`.

## 5. Statistical parameters (defaults)

| Parameter | Value | Note |
|---|---|---|
| CI level | 95% | BCa primary, percentile alongside |
| B (item-level) | 10,000 | cheap: pure-Python recompute |
| B (holistic, M5) | 1,000 | pipeline re-run per resample; ConEL-2 makes this minutes, not hours |
| Order permutations K | 40 | seeded; M3/M5, sanity on M4 |
| Threshold grid | 0.65–0.95 step 0.05 | resolver similarity threshold; reuses the sweep's existing points (0.65/0.75/0.80/0.85/0.90) and extends just past the 0.90 operating point |
| RNG | `random.Random(seed)`, fixed seed | same seed → byte-identical intervals |

## 6. Per-milestone deliverables

| Milestone | Bootstrap CI | Paired delta | Threshold curve | Order-permutation |
|---|---|---|---|---|
| **M2** Inspec `f1-at-k` | ✓ f1@5/10/15 (macro) | — | — | sanity (expect invariant) |
| **M3** ConEL-2 + ECB+ `clustering` | ✓ b3-f1 per config | ✓ **nn@0.90 − exact-label** (CI on delta + P(Δ>0)) | ✓ b3-f1 vs threshold, CI band | ✓ K orderings |
| **M4** 6 TExEval-2 golds `edge-f1` | ✓ P/R/F1 per gold, corpus-doc resample (glossary fixed) | — | — | sanity: expect ~0 spread → validates glossary-first design |
| **M5** ConEL-2 intrinsic (holistic) | ✓ redundancy / coherence / hierarchy-sanity (B=1000) | — | ✓ redundancy↔coherence tradeoff vs threshold (operating-point defense) | ✓ K orderings |

The M3 paired delta is the headline, and it must be computed at the **pre-registered
operating point nn@0.90** (chosen independently in M5) versus exact-label — **not** the
post-hoc best threshold. On b3-f1, nn@0.90 scores 0.9491 and exact-label 0.9386, a
delta of ≈ **+0.011** on 58 conversations — a razor-thin margin that motivates the whole
exercise. Comparing exact-label against the maximum over thresholds (0.9620, which sits
at 0.80, not the operating point) would be selection bias; the defensible claim uses the
threshold we committed to before this analysis. The threshold curve separately
characterizes sensitivity — and will honestly show that 0.90 is not b3-f1-optimal for
M3 (it was chosen for M5's redundancy↔coherence tradeoff, a different objective). The
deliverable states whether the 95% CI on the nn@0.90 − exact-label delta excludes 0, in
whichever direction it falls.

## 7. Testing strategy

- **Kernels** (`test_resample.py`): each of the three kernels on a hand-built toy bundle
  with hand-computed expected aggregates; multiplicity behavior asserted (mention-pooled
  respects duplicates; edge-set-pooled collapses them).
- **Equivalence** (critical): for each opted-in metric, `aggregate(emit_records(full))`
  == `evaluate(full)` exactly, on a toy pipeline run with the hashing embedder.
- **Intervals** (`test_intervals.py`): percentile CI against a synthetic resample list
  with an exactly computable interval; BCa against a small fixed sample with
  hand-verified `z0`/`a`; degenerate zero-variance → zero-width interval; `paired_delta`
  on a toy where the sign and `prob_positive` are known by construction.
- **Permutation** (`test_permutation.py`): order-invariant toy metric → spread 0;
  order-sensitive toy → spread > 0.
- **Determinism**: same seed → identical `interval-report.json`; different seed →
  intervals within tolerance of each other (stability), same point estimate.
- **CLI smoke**: run against the walking-skeleton config into `tmp_path`, assert the
  report shape (keys, method labels, level).
- No model downloads: every test uses toy fixtures and the hashing embedder; kernels and
  interval math are pure. No `@pytest.mark.ml` test is required (nothing here needs the
  real embedder — determinism makes toy embeddings sufficient).
- Existing suite (413 tests) passes unchanged.

## 8. Output artifacts

- `reports/<milestone>/interval-report.json` (gitignored, regenerable): per metric —
  point estimate, BCa CI, percentile CI, method flags, B, seed; per paired claim —
  delta, CI, `prob_positive`; per threshold grid — point + CI at each threshold; per
  permutation — spread stats.
- `docs/results/2026-07-14-interval-analysis.md` (**committed**, citable): the headline
  numbers with intervals, the paired-delta verdict in prose, the sensitivity and
  permutation tables, the exact CLI commands that regenerate every number, and the
  explicit statement of the edge-set-pooled bootstrap semantics. This is the durable
  artifact; the JSON is its regenerable source.

## 9. Success criteria

1. **Equivalence holds**: every opted-in metric's `aggregate(emit_records(full))` equals
   its `evaluate(full)` — the interval is centered on the published number.
2. **Reproducible**: a fixed seed produces byte-identical `interval-report.json` across
   runs; the results doc's numbers are regenerable by the documented commands.
3. **Headline claim adjudicated**: the M3 paired-delta CI and `prob_positive` are
   computed and reported — "beats" is either defensible (CI excludes 0) or honestly
   qualified.
4. **Full matrix covered**: every cell in the §6 table is produced.
5. **No regression**: full existing suite passes unchanged; ruff clean; no existing
   adapter, config, port method, or recorded result modified.

## 10. Explicitly deferred

- Multiple training seeds — inapplicable (deterministic pipeline); documented as the
  reason "seeds" is not the instrument.
- Multiple-comparison correction across the 6 M4 golds — notable, optional; the results
  doc may mention it but does not apply it.
- Bayesian / credible intervals; effect sizes beyond the paired delta.
- Track 2 (second M5 corpus) and Track 3 (excluded-method baseline) — separate specs.
- Wiring the stats layer into `Engine`/public API — this is a benchmark-analysis tool,
  not a consumer feature.
