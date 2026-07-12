# lattice M4 — Hierarchy Track Design Spec

**Date:** 2026-07-12
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` §13 milestone 4
**Status:** approved design, pre-plan

Milestone 4 of the parent spec: `RelationInducer` adapters (Hearst patterns,
compound head-modifier) evaluated against SemEval-2016 Task 13 (TExEval-2)
with edge precision/recall/F1.

## 1. Goal

Give the engine its first real `IS_A` edges. Two evidence sources, each an
adapter behind the existing `RelationInducer` port: **string structure**
(compound head-modifier: "olive oil" IS_A "oil") and **corpus evidence**
(Hearst-style lexico-syntactic patterns over running text). The benchmark
question the sweep answers: does corpus evidence add anything over string
structure — the strongest published TExEval-2 English system was the
string-compositionality baseline on 4 of 6 golds (task paper Table 3).

## 2. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Evaluation corpus | Wikipedia summaries, one per gold term | TExEval-2 ships no corpus (terms + gold edges only); participants supplied their own. Definitional summaries are copula-rich ("Chocos is a breakfast cereal…" directly yields the gold edge chocos→breakfast cereal). User decision 2026-07-11. |
| Benchmark scope | All six English golds | env-eurovoc, food (combined), food-wordnet, science (combined), science-eurovoc, science-wordnet — full comparability with the task paper's Table 3. User decision 2026-07-11. |
| Induction locality | Glossary-first stream, intra-document induction | Dataset yields a terminology glossary as document 0 (all terms co-resident), so the compound closure is intra-document and the port stays stateless. Rejected: ConceptStore lookup (order-dependent edges, quasi-stateful port); harness-side taxonomy construction (bypasses the port M4 exists to exercise). |
| Concept universe | The gold's `.terms` file exactly | Matches the official protocol (participants received only the term list). Gold edges with endpoints outside the term list are unreachable for every system — the published numbers carry the same handicap (see §5 recall ceilings). |
| Segmenter | `block` | Two food terms contain `". "` ("st. honoré cake", "st. louis-style pizza") — the sentence segmenter would split them mid-term, making them unmatchable by the gazetteer in any document. Block keeps the glossary a single unit (M3 precedent) and costs Hearst nothing: the connector between anchor spans must match exactly, so locality emerges from the patterns themselves (a cross-sentence `". And other"` connector fails the match). |
| Embedder | `hashing` | Embeddings are irrelevant to this protocol (exact-label resolution, string/pattern induction). Sweeps run model-free. |
| New dependencies | None | Anchored pattern matching needs no NP chunker: gazetteer mention spans are the NPs. `pyproject.toml` stays frozen. |

## 3. Core changes

**None.** `Relation`, `GraphSnapshot`, the `RelationInducer` port, the
snapshot-level `Metric` port, and the in-memory `GraphIntegrator` (which
already dedupes relations by `(type, source_id, target_id)`) all exist from
M1. M4 is adapters + a fetch script + configs only.

## 4. New adapters

### 4.1 RelationInducer `"compound"`

Constructor: `(longest_only: bool = True)`.

Per document: collect the distinct resolved concepts (`resolution.concept`),
keyed by label. For each multiword label, walk its word-suffixes from longest
to shortest ("extra virgin olive oil" → "virgin olive oil", "olive oil",
"oil"). If a suffix is the label of another concept resolved in the same
document, emit `Relation(type="IS_A", source_id=compound.id,
target_id=suffix_concept.id, confidence=1.0, provenance=document.id)`.
With `longest_only=True` (default) stop at the first (longest) matching
suffix — the direct parent; with `False` emit every matching suffix.
Suffixes are whole-word (split on spaces), never substrings — "pineapple"
does not yield "apple". No self-edges (a suffix equal to the full label is
impossible since suffix walking starts at one word shorter).

On the glossary document (§4.4) every gold term is co-resident, so this
adapter computes the complete compound closure of the term list there —
functionally the task paper's string baseline B, refined: suffix-only
(English head-final), where B linked on prefix **or** suffix.

### 4.2 RelationInducer `"hearst"`

Constructor: `(patterns: list[str] | None = None, copula: bool = True)`.
`patterns=None` means the full built-in set.

Anchored pattern matching — the resolved mentions are the NPs, no chunking:

1. Group this document's resolutions by `mention.mention.unit_id`; within a
   unit, order mentions by span start. Overlapping mention spans: keep the
   longer (gazetteer emits longest-match, but resolutions may still contain
   nested matches if `longest_only` semantics change — be defensive).
2. For each ordered pair of mentions (x before y or y before x per pattern
   direction) in the same unit, match the **connector text between the two
   spans** (and immediately around them where the pattern requires) against
   the pattern set.
3. Coordination walking: after a match `Y such as X₁`, continue consuming
   following mentions separated only by `,` / `, and` / `, or` / `and` /
   `or` — each Xᵢ yields an edge Xᵢ IS_A Y.

Built-in patterns (Hearst 1992), with NPy the hypernym and NPx the hyponym;
connector regexes are case-insensitive and tolerate an optional comma:

| name | surface shape |
|---|---|
| `such-as` | NPy `, such as` NPx / NPy `such as` NPx |
| `such-np-as` | `such` NPy `as` NPx |
| `including` | NPy `, including` NPx |
| `especially` | NPy `, especially` NPx |
| `and-other` | NPx `and other` NPy |
| `or-other` | NPx `or other` NPy |
| `copula` | NPx `is a`/`is an`/`are a`/`are an`/`is a kind of`/`is a type of` NPy (enabled by `copula=True`) |

The connector between the two anchor spans must match the pattern **exactly**
(allowing the article and whitespace variants baked into each regex) — any
other intervening text kills the match. This trades recall for precision and
is what keeps the adapter regex-only. Emitted edges:
`Relation(type="IS_A", source_id=x.concept.id, target_id=y.concept.id,
confidence=1.0, provenance=document.id)`; skip x.concept.id ==
y.concept.id (a term next to itself is not evidence).

### 4.3 RelationInducer `"union"`

Constructor: `(members: list[dict])` where each member is
`{"name": str, "params": dict}` (params optional). Instantiates each member
from the registry at construction time and concatenates their `induce`
outputs in member order. The graph integrator dedupes. Members that need
shared-dep injection are out of scope (none of the M4 members do); the
adapter documents this limitation and raises if a member's constructor
requires an argument not in its params.

### 4.4 Dataset `"taxonomy"`

Constructor: `(root: str, gold: str, limit: int | None = None)` where `gold`
is one of the six converted gold keys (§5). Reads
`{root}/{gold}/documents.jsonl` and `{root}/{gold}/gold_edges.jsonl`.

`documents()` yields, in file order:
1. Document 0: `id=f"{gold}:glossary"`, `kind="terminology"`, text = every
   term one-per-line (a single unit under the block segmenter — terms
   contain no blank lines), `timestamp=0.0`.
2. One document per term with a usable Wikipedia summary:
   `id=f"{gold}:{slug}"`, `kind="article"`, text = the summary extract,
   ordered by term, `timestamp` = 1-based position.

`limit` truncates the per-term documents (never the glossary) for smoke
tests. `ground_truth()` returns
`{"is_a_edges": [[hypo, hyper], ...], "terms": [...]}` — lowercased,
deduped, exactly as converted (§5).

### 4.5 Extractor `"gazetteer"`

Constructor: `(root: str, gold: str)` — loads the same gold's term list
(from `{root}/{gold}/terms.txt`, §5) so extractor and dataset stay paired,
M3 gold-mentions style, including the FileNotFoundError with a
run-the-fetch-script hint.

Per unit: case-insensitive, whole-word, longest-match dictionary matching of
the term list against `unit.text`. Implementation: one compiled alternation
regex with terms sorted longest-first and `re.escape`d, bounded by
`(?<!\w)…(?!\w)` (terms can contain hyphens and punctuation, so `\b` is
wrong at non-word-char boundaries). Overlaps: a match consumes its span;
scanning resumes after it (`finditer` on the alternation gives this for
free). Emits `Mention(surface=matched_text, unit_id, span, context=±40
chars)`; `head`/`lemma` stay empty.

### 4.6 Metric `"edge-f1"`

Registered under the snapshot-level `Metric` port (first real use since the
walking skeleton's `label-f1`).

`evaluate(snapshot, ground_truth)`: build `id → label` from
`snapshot.concepts`; collect `{(label(source), label(target))}` for
relations with `type == "IS_A"` (labels are already lowercase via
exact-label; lowercase again defensively); compare as sets against
`{tuple(edge) for edge in ground_truth["is_a_edges"]}`. Return
`{"precision", "recall", "f1", "predicted_edges", "gold_edges"}` (counts as
floats). Zero-denominator conventions match `label-f1`: empty predicted →
P=0, empty gold → R=0, P+R=0 → F=0.

## 5. Fetch script

`scripts/fetch_texeval.py` — stdlib only, M3 conventions (argparse `--root`
default `data`, checksums record-only, never committed; `data/` is
gitignored). macOS note: run with
`export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())")`.

1. Download `texeval-2_testdata_1.2.tar.gz` (alt.qcri.org, URL in script)
   to `data/texeval/` if absent; extract in place.
2. For each of the six English golds — key ↔ archive file:

   | key | terms file | taxo file | paper Table 3 row |
   |---|---|---|---|
   | `env-eurovoc` | `environment_eurovoc_en.terms` | `environment_eurovoc_en.taxo` | Environment/Eurovoc |
   | `food` | `food_en.terms` | `food_en.taxo` | Food/Combined |
   | `food-wordnet` | `food_wordnet_en.terms` | `food_wordnet_en.taxo` | Food/WordNet |
   | `science` | `science_en.terms` | `science_en.taxo` | Science/Combined |
   | `science-eurovoc` | `science_eurovoc_en.terms` | `science_eurovoc_en.taxo` | Science/Eurovoc |
   | `science-wordnet` | `science_wordnet_en.terms` | `science_wordnet_en.taxo` | Science/WordNet |

3. **Terms:** lowercase, strip, dedupe preserving first occurrence
   (`food_en.terms` has 1555 lines but 1549 unique lowercased terms).
   Write `{root}/texeval/{key}/terms.txt`, one per line.
4. **Gold edges:** from the `.taxo` file (`id⇥term⇥hypernym`), lowercase
   both endpoints, dedupe (food-wordnet has 43 duplicate edges,
   science-wordnet 11). Keep edges even when an endpoint is missing from
   the term list — the official gold is the full `.taxo` file. Write
   `{root}/texeval/{key}/gold_edges.jsonl` (one `[hypo, hyper]` pair per
   line) and print the **recall ceiling** (fraction of unique gold edges
   with both endpoints in the term list). Corpus-verified (2026-07-12):
   env-eurovoc 260/261 (0.996), food 1355/1587 (0.854), food-wordnet
   1533/1533 (1.0), science 464/465 (0.998), science-eurovoc 124/124
   (1.0), science-wordnet 382/441 (0.866).
5. **Wikipedia summaries:** for each unique term across all six lists
   (3,897), GET
   `https://en.wikipedia.org/api/rest_v1/page/summary/{quoted term with
   spaces→underscores}` with a descriptive User-Agent; cache each raw JSON
   response in `{root}/texeval/wiki-cache/{sha256(term)}.json` (fetch skips
   cached terms, so reruns are cheap and resumable). HTTP 404 and responses
   with `type != "standard"` (disambiguation etc.) → term gets no article
   document (it still becomes a concept via the glossary). Extract the
   `extract` field; skip if empty. Politeness: ~50ms sleep between uncached
   requests.
6. **Documents:** write `{root}/texeval/{key}/documents.jsonl`: first the
   glossary record (`{"id": "{key}:glossary", "kind": "terminology",
   "text": terms joined by "\n"}`), then one
   `{"id": "{key}:{slug}", "kind": "article", "text": extract, "term":
   term}` per term with a usable summary, in term order (slug = term with
   spaces→`-`, non-alphanumerics dropped; collisions get a numeric suffix).
7. `CHECKSUMS` per gold directory over the emitted files, record-only.

## 6. Configs + sweep

Six sweep configs `configs/m4-<key>-sweep.toml`, identical apart from
`root`-relative `gold`. Base row:

```toml
[base.segmenter]        name = "block"
[base.extractor]        name = "gazetteer"
[base.extractor.params] root = "data/texeval"
                        gold = "env-eurovoc"
[base.scorer]           name = "passthrough"
[base.resolver]         name = "exact-label"
[base.relation_inducer] name = "compound"
[base.graph_integrator] name = "in-memory"
[base.embedder]         name = "hashing"
[base.concept_store]    name = "in-memory"
[base.run]              on_error = "fail"
                        seed = 0
[base.dataset]          name = "taxonomy"
[base.dataset.params]   root = "data/texeval"
                        gold = "env-eurovoc"
[[base.metrics]]        name = "edge-f1"

[axes]
relation_inducer = [
  { name = "compound" },
  { name = "hearst" },
  { name = "union", params = { members = [
      { name = "hearst" }, { name = "compound" } ] } },
]
```

Run: `python -m lattice.harness --sweep configs/m4-<key>-sweep.toml
reports/m4-<key>` (the `--sweep` flag on `lattice.harness`; the
`lattice.harness.sweep` module path is a silent no-op — M2b lesson).
18 rows total (6 golds × 3 inducers). Reports land in gitignored
`reports/`.

## 7. Error handling

- Missing converted data → `FileNotFoundError` naming
  `scripts/fetch_texeval.py` (M3 pattern), from both dataset and gazetteer.
- Dataset/extractor gold mismatch is structurally impossible to detect
  cheaply (both just read the same directory) — the pairing is by config;
  the e2e test pins it.
- `union` member name not in registry → the registry's normal KeyError
  surfaces at construction (fail fast, not mid-stream).
- Sweep runs with `on_error = "fail"`: any per-document exception fails the
  row loudly.

## 8. Testing strategy

- **Unit, compound:** suffix walking (longest-only vs all), no self-edge,
  single-word labels inert, multiword suffix must be whole-word
  ("pineapple" ↛ "apple"), only same-document concepts linked.
- **Unit, hearst:** one test per built-in pattern; coordination walking
  ("y such as x₁, x₂ and x₃" → 3 edges); intervening-text kills match;
  copula flag off drops copula edges; same-concept pair skipped;
  cross-unit pairs never matched.
- **Unit, union:** member instantiation with params, output concatenation,
  registry KeyError on unknown member.
- **Unit, gazetteer:** longest-match wins, case-insensitive, non-word
  boundaries (hyphenated neighbors don't match), punctuation-adjacent
  matches, FileNotFoundError hint.
- **Unit, edge-f1:** exact/empty/disjoint cases, non-IS_A relations
  ignored, counts reported, zero-denominator conventions.
- **Unit, taxonomy dataset:** glossary first, timestamps ordered, limit
  semantics, ground-truth shape.
- **Fixtures:** `tests/fixtures/mini_texeval/<key>/…` — a hand-built
  3-term gold with one compound edge and one Hearst edge in a fake summary;
  no network anywhere in tests.
- **e2e:** `tests/harness/test_m4_e2e.py` — sweep a mini config over the
  fixture, assert all three rows run error-free and union f1 ≥
  max(member f1) on the fixture.
- **Fetch script:** `tests/scripts/test_fetch_texeval.py` — pure
  conversion functions (term normalization/dedupe, taxo→edges dedupe,
  summary-response filtering, slugging) against inline samples; no
  network.

## 9. Success criteria

1. **Mechanical:** all 18 sweep rows complete with zero errors; full test
   suite + ruff clean.
2. **Headline question:** `union` edge-F1 ≥ each individual member's on a
   majority of the six golds — corpus evidence (Hearst) must add recall to
   string structure (compound) without collapsing precision.
3. **Published band (task paper Table 3, English F-score):** the best
   lattice row per gold lands within or above the participant band —
   env-eurovoc 0.17–0.30, food 0.09–0.28, food-wordnet 0.21–0.36, science
   0.15–0.39, science-eurovoc 0.17–0.31, science-wordnet 0.24–0.38 — where
   the top of each band was usually the string baseline B (best on 4/6
   golds, avg F 0.33). Our compound row is B refined (suffix-only), so
   landing near B is the expectation; falling far below the band floor on
   any gold demands adjudication before close.
4. Recall ceilings (§5.4) are quoted next to results in the ledger so
   low recall on `food` (max 0.854) and `science-wordnet` (max 0.866) is
   read correctly.

## 10. Explicitly deferred

- Non-English TExEval subtasks (Dutch/French/Italian) — pipeline is
  English-only.
- Structural taxonomy measures (cycles, connectedness, Cumulative F&M) —
  parent spec names edge P/R/F only; structural sanity belongs to M5's
  intrinsic harness.
- Co-occurrence inducer benchmark row — it emits `CO_OCCURS`, which
  edge-f1 ignores by design; including it would score a flat 0.
- Confidence calibration (per-pattern confidence values) — all edges 1.0
  until something consumes confidence.
- Cross-document induction via ConceptStore — rejected in §2; revisit only
  if a streaming consumer needs incremental taxonomy growth.
