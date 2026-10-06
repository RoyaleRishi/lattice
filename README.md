# lattice

**A concept-memory engine.** lattice ingests a stream of documents — LLM
session transcripts, notes, any unit of text — and maintains an accreting
concept graph that carries concept identity **across** documents rather than
re-extracting keyphrases per document.

**What it does today, measured.** It collapses repeated and
morphologically-varied surface forms into one concept, induces `IS_A`
relations from the text, and resumes exactly from a save file. Against raw
string matching it recovers gold mention clusters better on both identity
benchmarks — B³ F1 0.9491 vs 0.9386 on ConEL-2, 0.6255 vs 0.6084 on ECB+ —
though only the ECB+ margin survives a family-wise correction. The gain is
small and it is mostly morphology: at the shipped operating point the
embedding resolver makes 7 merges beyond exact string matching on ConEL-2's
452 mentions, and a free Snowball stemmer reproduces 6 of them. Six of the
seven are singular/plural pairs ("ocean"/"oceans"); the seventh, and the only
one the stemmer misses, is `favorite color` / `favourite colour`. Rerun that
count with `scripts/m3_merge_ladder_check.py`.

**What it does not do yet.** Merge synonyms that share no string. On the
shipped `standard` profile, "vector store" and "vector database" remain two
concepts — their MiniLM cosine is 0.681, well under the 0.90 resolution
threshold. That merge is the roadmap target, not a current capability, and
lowering the threshold is not the fix: it costs coherence fast (see
Integration below). The fix is contextual resolution — embedding a mention in
its surrounding text instead of embedding the bare surface. `Mention.context`
is already populated by every extractor and read by nothing.

Every algorithmic stage is a swappable adapter behind a port. In each
benchmark below the adapter *under test* is the one **`standard`** ships,
though the surrounding harness config is the benchmark's own rather than the
profile verbatim. None of it is evidence about `lite`, whose resolver
threshold was never calibrated for its embedder — see Profiles.

## Install

```bash
uv add lattice            # core: dependency-light "lite" profile
uv add "lattice[ml]"      # + spaCy & sentence-transformers for "standard"
uv run python scripts/fetch_models.py   # one-time model download (standard)
```

## Quickstart

```python
from lattice import Engine

engine = Engine()  # "lite" profile — dependency-free; see the toggle below

engine.ingest("Olive oil is a fat prized in Mediterranean cooking.")
engine.ingest("Mediterranean groves grow olive trees for oil.")

view = engine.view()
olive = view.find_concept("olive")

engine.save("memory.json")            # versioned JSON, survives restarts
restored = Engine.load("memory.json")
restored.ingest("Olive presses yield fresh oil each autumn.")
```

`Engine()` defaults to the **lite** profile: the same pipeline topology as the
real thing, with a toy tokenizer and hashing embedder — instant,
dependency-free, right for smoke tests and CI. Note its limits: it only sees
single words of ≥ 4 letters ("olive", not "olive oil").

**For real use, flip one switch:**

```
engine = Engine(profile="standard")
```

## Profiles

| stage | lite | standard | what picked `standard`'s |
|---|---|---|---|
| extractor | token (words ≥ 4 chars) | spaCy noun chunks | M2 spec: real noun phrases. Held fixed as the base of every sweep — **never itself swept** |
| embedder | hashing trigrams | all-MiniLM-L6-v2 | M2/M3 spec. Likewise held fixed, **never itself swept** |
| scorer | embedding-cosine | embedding-cosine | M2b scorer sweep: best f1 at every k of four scorers (Salience below) |
| resolver | embedding-nn @ 0.90 | embedding-nn @ 0.90 | M5 resolver sweep, on the redundancy↔coherence tradeoff — *not* on b3-f1, which peaks elsewhere (Integration below) |
| relations | hearst + compound union | hearst + compound union | M4 relation-inducer sweep: union ≥ both members on 6/6 golds (Hierarchy below) |

Only the last three stages were ever swept; `scorer`, `resolver` and
`relation_inducer` are the only axes any config in `configs/` varies. The
extractor and embedder are design decisions recorded in `docs/`, held
constant underneath every comparison.

Both profiles share one topology — switching changes quality, never behavior
shape. Full control: `Engine.from_config(path_or_dict)` with the same TOML
schema the experiment harness uses.

**The right-hand column is about `standard` only.** `lite` inherits
`embedding-nn @ 0.90` from `standard`, but 0.90 was calibrated in MiniLM
cosine space, and `lite` swaps in a hashing-trigram embedder whose similarity
geometry is unrelated — a trigram-overlap cosine of 0.90 does not mean what a
MiniLM cosine of 0.90 means. No sweep has ever been run on the `lite` stack.
Treat it as a topology-preserving smoke-test profile, not as a scaled-down
version of the measured system.

## Benchmark evidence

Unless stated otherwise, the numbers below are point estimates from the
artifacts under `reports/`, regenerated 2026-08-01 at seed 0. Two exceptions,
both marked where they appear: the 2016 TExEval-2 participant bands are
transcribed from the task paper via
`docs/architecture/2026-07-12-m4-hierarchy-design.md:299-305`, and the order-sensitivity
subsection is a permutation study at **seed 1**. Provenance, before/after
diffs and the full interval tables: `docs/results/2026-07-31-post-fix.md`.

### Salience — Inspec test split, **n = 500 documents**

Ranking spaCy noun-chunk candidates; `f1-at-k` macro-averaged over documents.

| scorer | f1@5 | f1@10 | f1@15 |
|---|---|---|---|
| **embedding-cosine** (default) | **0.2986** | **0.3544** | **0.3555** |
| mderank | 0.2748 | 0.3276 | 0.3331 |
| hcuke | 0.2572 | 0.3029 | 0.3152 |
| frequency | 0.1785 | 0.2413 | 0.2692 |

`embedding-cosine` f1@10 = 0.3544, BCa 95 % CI [0.3396, 0.3692] (B = 10000,
n = 500, brackets its estimate).

**This is an in-project comparison only.** `frequency` is what its own
docstring calls it — a "trivial walking-skeleton scorer", surface count over
max surface count. Beating it is a floor check, not a result. No classical
unsupervised baseline (TF-IDF, TextRank, YAKE) is implemented in this
repository, so the table places `embedding-cosine` against three other lattice
adapters and against nothing from the keyphrase-extraction literature.
`hcuke`'s row is post-fix: the adapter was broken (a product over factors, one
of which could go negative, inverted the ranking) and sat below `frequency` at
every k; fixed, it clears `frequency` at every k and still finishes last of
the three embedding-based scorers.

### Identity — ConEL-2 (**n = 58 conversations / 452 mentions**) and ECB+ (**n = 206 documents / 2054 mentions**)

Gold mentions in, B³ against gold clusters. Corpus sizes are the
`corpus_census` block of
`reports/intervals/analysis/m3-merge-ladder-check.json`, counted through the
same `mention-clusters` reader the runs use. Three resolvers, one ladder:

| resolver | ConEL-2 b3-f1 | ECB+ b3-f1 |
|---|---|---|
| exact-label | 0.9386 | 0.6084 |
| stemmed-label | **0.9637** | 0.6154 |
| embedding-nn @ 0.90 (default) | 0.9491 | **0.6255** |

Paired deltas, same document draws in both arms (subsample, B = 10000,
seed 0), with Holm–Bonferroni at α = 0.05 over all six comparisons:

| corpus | comparison | Δ b3-f1 | 95 % CI | Holm |
|---|---|---|---|---|
| ConEL-2 | stemmed − exact | +0.0251 | [+0.0093, +0.0311] | survives |
| ConEL-2 | nn@0.90 − exact | +0.0105 | [+0.0023, +0.0160] | **fails** |
| ConEL-2 | nn@0.90 − stemmed | −0.0146 | [−0.0220, −0.0004] | **fails** |
| ECB+ | nn@0.90 − exact | +0.0171 | [+0.0091, +0.0215] | survives |
| ECB+ | stemmed − exact | +0.0070 | [+0.0026, +0.0102] | survives |
| ECB+ | nn@0.90 − stemmed | +0.0101 | [+0.0030, +0.0148] | survives |

Reading, in three parts:

- **Morphological normalization beats raw string matching on both corpora**,
  and that survives family-wise control. This is the solid part of the
  identity result.
- **The embedding's advantage over the free stemmer reverses by corpus.** It
  wins on ECB+ by 0.0101 b3-f1 and loses on ConEL-2 by 0.0146. `stemmed-label`
  uses the Snowball stemmer, already a core dependency: no model download, no
  embedder call, no `[ml]` extra. And ConEL-2 — conversational entity linking
  — is the corpus the M5 design spec calls "closest to the LLM-session
  framing", i.e. the one most like lattice's own stated use case. The default
  resolver is behind the free baseline there.
- **Neither ConEL-2 claim involving `embedding-nn` survives the multiplicity
  correction.** Both are stated above with that qualification and must not be
  quoted without it. The honest reading is that 58 conversations cannot
  separate these three resolvers at FWER 0.05, while ECB+'s 206 documents can.
  `stemmed-label` is registered and selectable today; it is not the default,
  because the evidence does not order the two consistently.

Two further cautions. Earlier versions of this README quoted **0.962** and
**0.643**. Those are maxima over the threshold grid — 0.9620 at threshold 0.80
on ConEL-2, 0.6428 at 0.75 on ECB+ — not the shipped 0.90, and not the same
threshold as each other. This project's own intervals spec names that exact
comparison as selection bias (`docs/architecture/2026-07-14-statistical-intervals-design.md`
§6) and requires the pre-registered operating point; the old README did not
follow it. The shipped numbers are 0.9491 and 0.6255, and 0.9620 is in any
case now below `stemmed-label`'s threshold-free 0.9637. Separately: the
*marginal* B³ bands on ECB+ carry `brackets_estimate: false` and are corpus-size
sensitivity ranges, not confidence intervals. The paired deltas above are
well-posed because the size dependence is shared by both arms and largely
cancels in the difference.

### Hierarchy — TExEval-2, 6 English golds

| gold | compound | hearst | **union** | 2016 participant band | union clears band top? |
|---|---|---|---|---|---|
| food | 0.2688 | 0.1307 | **0.3233** | 0.09–0.28 | **yes**, +0.0433 |
| food-wordnet | 0.3462 | 0.1165 | **0.3881** | 0.21–0.36 | **yes**, +0.0281 |
| science | 0.3645 | 0.0335 | **0.3725** | 0.15–0.39 | no, −0.0175 |
| science-wordnet | 0.3119 | 0.0090 | **0.3169** | 0.24–0.38 | no, −0.0631 |
| env-eurovoc | 0.2950 | 0.0226 | **0.2991** | 0.17–0.30 | no, −0.0009 |
| science-eurovoc | 0.2222 | 0.0157 | **0.2338** | 0.17–0.31 | no, −0.0762 |

`edge-f1`. The union beats both of its members on 6/6 golds and clears the top
of the published band on 2/6. The band column is the only thing on this page
not from `reports/`: it is the participant range from the 2016 task paper's
Table 3 (English F-score), transcribed at
`docs/architecture/2026-07-12-m4-hierarchy-design.md:299-305`.

**The band comparison is not like-for-like, and both departures favour
lattice:**

1. **TExEval-2 ships no corpus** — only term lists and gold edges. The 2016
   participants supplied their own text; lattice supplies one Wikipedia summary
   per gold term. Definitional summaries are copula-rich ("Chocos *is a*
   breakfast cereal…" directly yields the gold edge), which is structurally
   easier than running prose. "Above the published band" is therefore not
   "beats the 2016 state of the art", and should not be read as it.
2. **`compound` is the task paper's own string baseline B, refined**
   (suffix-only, where B linked on prefix or suffix) — and B usually *set* the
   top of each band. So part of what is being compared is a narrowed 2016
   baseline against a field that includes that baseline's unrefined score.
   `compound` alone stays below the band top on all six golds; the two
   band-clearings are a joint product of it and Hearst's marginal contribution
   (+0.0545 on food, +0.0419 on food-wordnet).

The `food` and `food-wordnet` interval bands are likewise not confidence
intervals: `edge-f1` unions per-document edge sets, so the statistic is
monotone in distinct-document count and the resampled value sits below the
full-corpus estimate by construction. They are reported as size-sensitivity
ranges. The same condition catches `science`'s `predicted_edges` cell, and the
cells carrying it are not enumerated here — `brackets_estimate: false` in
`reports/intervals/*/interval-report.json` is the authority.

### Integration — ConEL-2 intrinsic, **n = 58 conversations**

Redundancy against coherence across the resolver axis, all four arms measured
at the same redundancy threshold (0.60, below every point on the resolver
axis):

| resolver | duplicate-rate | coherence | concepts |
|---|---|---|---|
| exact-label | 0.7040 | 1.0000 † | 554 |
| embedding-nn @ 0.90 (default) | 0.6769 | 0.9311 | 523 |
| embedding-nn @ 0.75 | 0.5096 | 0.8338 | 418 |
| embedding-nn @ 0.65 | 0.2853 | 0.7667 | 347 |

† vacuous: `exact-label` never merges two distinct surfaces, so there are no
multi-surface concepts to score and coherence is 1.0 by definition.

**At the shipped operating point the resolver removes 3.8 % of the redundancy
`exact-label` leaves.** Same story on the other two corpora: ECB+ 0.7556 →
0.7257 (4.0 %), MultiWOZ 0.9299 → 0.9102 (2.1 %). An earlier version of this
README claimed `duplicate-rate 0.117 → 0.015`, a 7.7× reduction. That number
was measured with the redundancy metric's cosine threshold set *equal to* the
resolver's, which makes a cosine-flagged duplicate impossible by construction —
the resolver has already merged every pair that could clear the bar. It was a
tautology, not a measurement, and the metric's docstring now carries the
circularity warning.

What the table does show is the tradeoff's shape: the resolver *can* cut
redundancy substantially — 0.7040 → 0.2853 at threshold 0.65 — but it pays in
coherence (1.0000 → 0.7667) and folds 554 concepts down to 347. 0.90 was
chosen for that tradeoff, not for identity: on ConEL-2, b3-f1 peaks at
threshold 0.80 (0.9620) and 0.90 sits past the peak.

### Sensitivity to document order

Ingestion is a stream, so the same corpus in a different order can give a
different graph. K = 40 shuffled orderings, seed 1; the table is the observed
range (max − min) per key.

| entry | key | range over 40 orderings |
|---|---|---|
| M3 ConEL-2 nn@0.90 | b3-f1 | 3.3e-16 |
| M3 ConEL-2 nn@0.90 | b3-recall | 7.8e-16 |
| **M3 ECB+ nn@0.90** | **b3-f1** | **0.0037** |
| **M3 ECB+ nn@0.90** | **b3-recall** | **0.0055** |
| **M3 ECB+ nn@0.90** | **ari** | **0.0066** |
| M4, all six golds (glossary pinned) | every key | 0.0 |
| M5 ConEL-2 | concept-count / is-a-edges | 2 / 7 |
| M5 ConEL-2 | redundancy.duplicate-rate (metric threshold **0.80**) | 0.0124 |

**Identity is order-invariant on ConEL-2 and is not on ECB+**, and that
matters for how hard the identity result above can be pushed, because ECB+
carries every surviving margin that involves the shipped resolver. The largest
of them, nn@0.90 − exact-label, is Δ = +0.0171 with CI lower bound +0.0091: an
order-induced b3-f1 range of 0.0037 on the same corpus is 22 % of that point
estimate and 41 % of that lower bound. The margin the corpus-reversal claim
actually rests on is smaller — nn@0.90 − stemmed-label, Δ = +0.0101, CI
[+0.0030, +0.0148] — and 0.0037 exceeds its CI lower bound outright. The
paired-delta construction resamples documents but holds one insertion order
fixed, so this variability is not inside those intervals; it sits alongside
them. Read the ECB+ margins as real but not much larger than the pipeline's
own order noise.

M4's exact 0.0 is narrower than it looks: those six runs set `fixed_prefix: 1`,
pinning the glossary document at stream position 0 and shuffling only the
remainder. That is the glossary-first design working as specified, not
invariance under a full shuffle. The M5 rows are the ordinary case — greedy
arrival-order merging decides which member of a similar pair survives, so the
accreted graph's *size* moves even where its clustering quality does not.

**Do not read the 0.0124 against the 0.6769 in the Integration table above.**
The order study runs `configs/m5-conel2-nn090.toml`, whose redundancy metric
threshold is 0.80; the Integration table runs the sweep config at 0.60. The
level 0.0124 was measured against is `duplicate-rate` **0.2524**
(`reports/intervals/m5-conel2-b50/interval-report.json`), so the order-induced
range is ~4.9 % of its own baseline, not the ~1.8 % a cross-table division
would suggest.

## What is not measured, and what is stale

Stated so a reader can tell where the evidence stops.

- **PromptRank.** A `promptrank` scorer adapter exists; none of its published
  numbers are quoted here as a result, and three artifacts behind them are
  stale. In
  `reports/intervals/promptrank/promptrank-paired-delta.json` the incumbent
  column is `embedding-cosine` at pre-fix values, so both arms are stale at
  f1@5 and f1@10 — **but not at f1@15**, where the incumbent
  `0.35545236519207857` is byte-identical to the current post-fix value in
  `reports/intervals/m2b/interval-report.json` (the stem-dedupe fix saturates
  at k = 15, `min(k, n_distinct_stems)`). The fragile row is the f1@15 one:
  its upper bound is **−8.5e-05**, i.e. 8.5e-05 *below* zero, and **only the
  PromptRank arm of it is unmeasured**, so re-measuring that one arm can flip
  its sign. Mind the direction: that row's Δ is
  **−0.0064** (promptrank 0.3490 against incumbent 0.3555), so PromptRank is
  the *worse* arm at f1@15 and what is fragile is a significantly-worse
  finding. `docs/results/2026-07-29-promptrank-baseline.md` states this
  correctly and carries a superseded banner, but
  `docs/results/2026-07-31-post-fix.md` §8 glosses the row as PromptRank's
  own "significantly better" verdict, which inverts it. Do not inherit that
  reading. `reports/m2-promptrank-sweep/`
  additionally still carries the pre-fix HCUKE row, and
  `reports/intervals/m2-promptrank/` predates the F1@K denominator fix that
  applies to every scorer. All three need a re-run, not an adjustment.
- **Three M5 holistic interval reports** (`reports/intervals/m5-conel2`,
  `-ecbplus`, `-multiwoz`) were built under the circular redundancy threshold
  and have not been regenerated. Nothing on this page quotes them.
- **No multiplicity correction outside the six M3 comparisons above.** The
  sweep tables and the remaining published intervals are per-comparison
  statements.
- **No external salience baseline**, as noted above.
- **Concept aliases are not persisted.** The merge decision is the engine's
  core artifact; `Concept` has no alias field and `save()` writes none.
- **The concept store is a brute-force linear scan** with a pure-Python
  cosine. numpy is deliberately not a core dependency, so a vectorized store
  belongs behind the `[ml]` extra as a separate `ConceptStore` adapter.

Specs and sweeps: `docs/` (start at
`docs/architecture/2026-07-05-lattice-architecture-design.md`). Results, including the
negative ones: `docs/results/`. Development setup, architecture conventions and
results discipline: [CONTRIBUTING.md](CONTRIBUTING.md).

## Persistence

`engine.save(path)` writes versioned JSON (`format_version: 2`) holding the
fully resolved config, the graph, the document counter, and the resolver's
private state. `Engine.load(path)`
rebuilds the engine and **resumes exactly**: processing A, B, save, load, C
equals processing A, B, C in one run (test-enforced). `load` reads only
v2; convert an old v1 save once with
`uv run python -m scripts.migrate_save old.json new.json`.

## API stability

Pre-1.0: the public contract is `lattice.__all__` — `Engine`, `GraphView`,
`Document`, `Concept`, `Relation`, `GraphDelta`, `GraphSnapshot`,
`__version__`. Minor versions may break it with a changelog note. Everything
below the top level is internal. Save files carry `format_version` and are
readable by any lattice that understands it.
