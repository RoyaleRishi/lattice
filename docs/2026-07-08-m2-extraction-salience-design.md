# lattice M2 — Extraction + Salience Track Design Spec

**Date:** 2026-07-08
**Status:** Approved design, pending spec review
**Parent:** `docs/2026-07-05-lattice-architecture-design.md` (§13 Milestone 2, §14 open questions)
**Baseline:** M1 walking skeleton complete at `6bb7e10` (120 tests, e2e harness green)

---

## 1. Goal

Replace the walking skeleton's trivial extraction/salience adapters with research-grade ones and
produce the project's first credible benchmark comparison: a reproducible sweep table scoring
competing salience algorithms on public keyphrase benchmarks.

## 2. Staging (decided)

One spec (this document), **two implementation plans**:

- **M2a — foundation + baseline:** heavy deps, noun-chunk extractor, sentence-transformer
  embedder, Inspec dataset, F1@k document metric, embedding-cosine baseline scorer, sweep
  runner. Exit: a reproducible sweep table with the cosine baseline on Inspec.
- **M2b — frontier scorers:** MDERank and HCUKE scorer adapters, compared against the baseline
  in the same sweep. M2b planning may react to M2a's real numbers.

**PromptRank is excluded from M2** (settles §14): the "no generative LLM on the critical path"
constraint's spirit is deterministic/self-hosted, which a local T5 scoring pass arguably meets,
but M2 already compares three scorers; a second model family isn't worth the dependency and
complexity cost before baseline numbers exist. Revisit after M2b — the `Scorer` port makes it a
pure addition.

## 3. Decisions log

| Decision | Choice |
|---|---|
| Staging | Two stages, one spec |
| Heavy deps | Full stack: sentence-transformers + torch + spaCy, in a uv optional group |
| Per-document evaluation | Enrich `GraphDelta` with `selected_mentions`; new `DocumentMetric` port |
| PromptRank (§14) | Excluded for M2, revisit post-M2b |
| Scorer machinery | All scorers consume the existing `Embedder` port (approach A) |
| Default embedding model (§14) | `all-MiniLM-L6-v2` (literature parity) |
| Multi-run state (M1 review note) | Fresh orchestrator per sweep config via the factory — no `reset()` reuse across configs |

## 4. Core change (one field)

`GraphDelta` gains:

```python
selected_mentions: tuple[ScoredMention, ...] = ()
```

Populated by the orchestrator with the scorer's selected output (pre-resolver), in scorer output
order. Backwards-compatible (defaulted). Rationale: keyphrase benchmarks are per-document; the
scorer's ranked selections are the unit under evaluation, and the accreting graph deliberately
destroys that view. The field doubles as per-document provenance for downstream consumers.

## 5. New port: `DocumentMetric`

```python
class DocumentMetric(ABC):
    """Harness port: scores per-document pipeline output against per-document
    ground truth. Complements the snapshot-level Metric port."""

    @abstractmethod
    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]: ...
```

- Registered/looked-up via the existing registry; harness `ExperimentConfig` grows
  `document_metrics: list[AdapterSpec] = []`; `RunReport.metrics` carries both families keyed by
  adapter name.
- Benchmark ground-truth shape: `{"keyphrases_by_document": {document_id: [phrase, ...]}}`.
  The snapshot `Metric` family and its `{"concept_labels": [...]}` shape are unchanged.

## 6. New adapters

### 6.1 Extractor `"noun-chunk"` (M2a)

spaCy `en_core_web_sm` noun chunks. Leading determiners/pronouns trimmed; `head`/`lemma` filled
from spaCy analysis; surfaces lowercased; spans indexed into the unit text (extractor contract).
Params: `max_tokens` (default 5, drop longer chunks), `model` (default `en_core_web_sm`).
The spaCy pipeline is loaded once per adapter instance with parser disabled where possible
(`noun_chunks` needs the tagger + parser; keep defaults minimal but correct).

### 6.2 Embedder `"sentence-transformer"` (M2a)

Wraps `sentence_transformers.SentenceTransformer`. Params: `model` (default `all-MiniLM-L6-v2`),
`batch_size` (default 32), `device` (default `"cpu"`). `dim` read from the loaded model.
Deterministic (inference only, no dropout). L2-normalizes outputs (matching the port's contract
as clarified in the M1 fix wave: empty input → zero vector, documented). Model name + version
are part of the stamped config (§7 reproducibility in the parent spec).

### 6.3 Scorer `"embedding-cosine"` (M2a)

The KeyBERT/SIFRank-style baseline: `salience = cosine(embed(candidate_surface),
embed(document_text))`, document text = units joined by `"\n"`. Candidates deduped by surface for
embedding (one embed call per unique surface + one for the document). Top-`top_k` unique surfaces
selected (default 10), ties lexicographic. Consumes only the injected `Embedder`.

### 6.4 Dataset `"inspec"` (M2a)

- Acquisition: `scripts/fetch_datasets.py` loads the Inspec benchmark (Hulth 2003) from the
  contemporary-standard `midas/inspec` HuggingFace dataset (proper train/validation/test splits)
  and converts it once to plain JSONL (`{"id", "text", "keyphrases"}` per line) under git-ignored
  `data/inspec/`, recording SHA-256 checksums of the emitted files. The `datasets` library lives
  in the `ml` group and is imported only by the fetch script — the adapter reads JSONL with the
  stdlib. Datasets are never committed.
- Adapter params: `root` (default `data/inspec`), `split` (default `"test"`), `limit`
  (optional int, for smoke runs). Yields one `Document` per abstract (`kind="abstract"`,
  `timestamp` = stable index order); `ground_truth()` returns the per-document gold keyphrases
  using the **uncontrolled** keyword set (the literature standard for Inspec), documented in
  the adapter.
- A 3-document mini-fixture (`tests/fixtures/mini_inspec/`) is committed so dataset-adapter and
  e2e tests run without downloads.

### 6.5 DocumentMetric `"f1-at-k"` (M2a)

Literature-standard evaluation: for each document, predicted = the top *k* unique surfaces of
`selected_mentions` ranked by salience descending, ties lexicographic (the metric sorts — no
ordering contract is imposed on scorers); gold = that document's keyphrases; both sides
Snowball-stemmed token-wise and compared as exact stemmed-phrase matches. Macro-averaged
precision/recall/F1 reported for each k. Params: `ks` (default `[5, 10, 15]`). Output keys:
`"precision@5"`, `"recall@5"`, `"f1@5"`, etc. Documents missing from ground truth are an error
(fail loudly, never silently shrink the eval set — parent spec §8 spirit).

Dependency: `snowballstemmer` (tiny, pure Python) as a regular dependency.

### 6.6 Scorer `"mderank"` (M2b)

MDERank (arXiv 2110.06651): for each candidate, mask all its occurrences in the document text
and re-embed; `salience = 1 - cosine(embed(original_doc), embed(masked_doc))` — candidates whose
removal moves the document embedding most are most salient. Implementation notes: mask token =
the model's mask/pad convention per paper (use `"[MASK]"` literal for MiniLM-family, documented);
one embed batch of `[original] + [masked_i]`; document truncation follows the embedder model's
limit (Inspec abstracts fit MiniLM's 256 tokens — assert and document). Params: `top_k`.

### 6.7 Scorer `"hcuke"` (M2b)

HCUKE (Knowledge-Based Systems 2024), hierarchical significance: document → unit → candidate.
Candidate significance combines (a) candidate–unit centrality, (b) unit–document centrality,
(c) candidate position/frequency features, per the paper's unsupervised formulation, all over
embeddings from the injected `Embedder`. Exact formula fidelity is a plan-time concern: the M2b
plan must cite the specific equations implemented and any deviations. Params: `top_k` plus the
paper's weighting hyperparameters with paper-default values.

## 7. Sweep runner (M2a)

`src/lattice/harness/sweep.py`:

- **SweepConfig** (pydantic, `extra="forbid"`): a full base `ExperimentConfig` mapping plus
  `axes: dict[port_name, list[AdapterSpec]]` — e.g. `axes.scorer = [{name="embedding-cosine"},
  {name="mderank"}]`. Loadable from TOML via the existing `load_config(path, model=...)`.
- **Expansion:** cartesian product of axes over the base config → ordered list of stamped
  `ExperimentConfig`s. Pure function, unit-tested.
- **Execution:** each config runs through the existing `run_experiment` with a **fresh
  orchestrator built by the factory** (settles reset-vs-reinstantiate: reinstantiate per config;
  stateful ports' `reset()` remains for intra-config use only).
- **Output:** `SweepReport` = list of `RunReport`s + a comparison table (rows = configs, columns
  = metric keys), emitted as JSON (machine) and a markdown table (human) to `reports/`
  (git-ignored) with the sweep spec stamped. `python -m lattice.harness.sweep <sweep.toml>`.

## 8. Packaging, hygiene, environment

- **uv optional group `ml`:** `sentence-transformers`, `spacy` (torch arrives transitively).
  Core dependencies gain only `snowballstemmer`. Default `uv sync` stays lean; `uv sync --group ml`
  opts in. `scripts/fetch-models.py` (or documented `spacy download`) fetches `en_core_web_sm`;
  model downloads never happen implicitly inside adapters at test time.
- **Tests:** model-dependent tests guard with `pytest.importorskip` + a marker (`ml`) and skip
  cleanly when the group isn't installed; all algorithmic logic (masking, stemming, F1
  arithmetic, sweep expansion, chunk trimming rules) is tested pure with fixtures. Existing
  contract suites (Extractor/Embedder/Scorer/Dataset) are reused by the new adapters —
  `DocumentMetric` gets its own contract suite.
- **Hygiene (from M1 final review):** add `ruff` config (line length matching existing style,
  default rule set) + `src/lattice/py.typed`; wire `uv run ruff check` into the dev loop.
- **Known machine quirk:** macOS UF_HIDDEN on venv `.pth` files recurs after `uv sync` —
  documented in `.superpowers/sdd/progress.md`; all plans must use
  `chflags nohidden .venv/lib/python*/site-packages/*.pth && uv run --no-sync ...`.

## 9. Error handling

- Dataset adapter: missing/corrupt data → explicit error naming the fetch script; checksum
  mismatch aborts.
- Embedder adapter: model not present → explicit error naming the fetch step (no silent
  downloads in tests; explicit opt-in for harness runs).
- DocumentMetric: document ids missing from ground truth → hard error (parent spec §8: eval sets
  never shrink silently).
- Everything else inherits the orchestrator's `on_error` policy unchanged.

## 10. Testing strategy

- Contract suites: new adapters pass their existing port contracts; new `DocumentMetricContract`
  (returns float dict; empty deltas handled; missing-doc error case).
- Pure-unit: chunk trimming, masking construction, stemmed matching, F1@k arithmetic on
  hand-computed fixtures, sweep cartesian expansion (order + stamp determinism).
- Integration (ml-marked): mini-Inspec fixture end-to-end through
  `run_experiment` with noun-chunk + sentence-transformer + cosine; sweep smoke over 2 configs.
- Reproducibility: identical sweep spec → byte-identical JSON report (asserts stamping +
  determinism).

## 11. Success criteria

- **M2a:** `python -m lattice.harness.sweep configs/m2a-baseline-sweep.toml` produces a
  reproducible table; cosine baseline F1@10 on Inspec test split lands in the published
  embedding-baseline sanity band (~0.25–0.35); default test suite green with and without the
  `ml` group installed.
- **M2b:** MDERank lands in its published Inspec band (paper Table 3: MDERank(BERT)
  F1@5/10/15 = 26.17/33.81/36.17; sanity band F1@10 ≈ 0.28–0.38 under the MiniLM embedder);
  one sweep table compares all scorers; HCUKE implementation documents its equations against
  the paper. *(Amended 2026-07-10: the original criterion "MDERank ≥ cosine baseline on
  Inspec, as the paper reports" misread the paper — Table 3/§5.3 report MDERank(BERT)
  underperforming EmbedRank(BERT) on Inspec specifically, because Inspec gold favors long
  phrases; MDERank's wins are on long-document datasets. Beating the cosine baseline on
  Inspec is therefore not required.)*

## 12. Explicitly deferred

- PromptRank adapter (post-M2b decision).
- SemEval-2010/2017, DUC-2001 datasets (add after Inspec proves the harness; pure adapter work).
- GPU support, embedding caches, long-document truncation strategies beyond MiniLM's window.
- Normalization/hierarchy tracks (M3/M4 per parent spec).
