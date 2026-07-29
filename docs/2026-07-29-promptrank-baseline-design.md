# Excluded-Method Baseline (PromptRank) — Design Spec (Credibility Track 3)

**Goal:** Re-admit the one keyphrase-salience method M2 deliberately excluded —
**PromptRank** (Kong et al., ACL 2023) — as a `Scorer` adapter, run it under
identical M2 conditions on Inspec, and report where it lands relative to the four
included scorers with Track-1-grade statistics, so the "we compared the frontier"
claim is not vulnerable to *"you left out the obvious generative baseline."*

**Status:** Design approved 2026-07-29. Third of three credibility sub-projects
(Track 1 = statistical intervals, shipped; Track 2 = M5 cross-corpus, shipped;
Track 3 = this).

---

## 1. Motivation

M2 compares four salience scorers — `frequency`, `embedding-cosine`, `mderank`,
`hcuke` — and claims to survey the keyphrase-extraction frontier. But it
**explicitly excluded PromptRank** ([m2 design §2](2026-07-08-m2-extraction-salience-design.md)):
the "no generative LLM on the critical path" constraint plus "a second model
family isn't worth the dependency and complexity cost before baseline numbers
exist. Revisit after M2b — the `Scorer` port makes it a pure addition." That
deferral was reasonable, but it leaves a credibility hole: PromptRank is the
strongest recent *unsupervised generative* method, and omitting it invites
*"you excluded it because it would win."*

Track 3 closes the hole by doing exactly what the M2 spec anticipated — adding
PromptRank as a pure `Scorer` addition — and reporting the result honestly
whichever way it falls. **PromptRank remains a benchmark-comparison entry only;
it is never wired into the shipped M6 engine**, so the "no generative model on
the critical path" property of the delivered pipeline is preserved. The point is
not to crown a winner but to remove the selection-bias objection.

## 2. Decisions (locked with the user, 2026-07-29)

| Decision | Choice | Rationale |
|---|---|---|
| Excluded method | **PromptRank** (ACL 2023) | The only method the repo ever names as "excluded"; the strongest omitted generative baseline. No textual support for any alternate referent (e.g. an LLM resolver). |
| Faithfulness | **Faithful T5 PromptRank** | The credibility purpose is to re-admit the *generative* method that was excluded *for being generative*; an embedding-only approximation would not be PromptRank and would not answer the objection. |
| Model checkpoint | **`t5-base`** (~220M) | The paper's checkpoint. Deterministic, CPU-runnable, cacheable via the sanctioned `fetch_models.py` path. |
| Dependency | **None added** — `transformers` used transitively | `transformers` is already resolved via `sentence-transformers` (uv.lock); `pyproject` stays frozen. Only a new *model* is downloaded, through the one sanctioned path. |
| Statistics | **Sweep + bootstrap CIs + paired delta** | Reuses Track 1's `f1-at-k` resampling. Reports F1@5/10/15 with CIs and a paired PromptRank-vs-strongest-incumbent delta — credibility-consistent with Tracks 1–2. |
| Dataset / split | **Inspec test (500 docs), seed 0** | Same `[base.*]` as the committed `configs/m2b-f1atk.toml` → apples-to-apples with the published M2 numbers. |
| Critical-path status | **Comparison-only** | Never added to the M6 engine defaults; preserves the "no generative LLM on the critical path" guarantee. |

## 3. Architecture

**One new `Scorer` adapter; no pipeline/harness source changes.** Like Track 2,
this is a near-pure addition: the composition root, sweep runner, `f1-at-k`
metric, and Track 1's entire `lattice.harness.stats` layer are reused unchanged.

The adapter, `promptrank`, implements the standard `Scorer` contract
`score(mentions, units) -> list[ScoredMention]` with the same
salience → rank (desc, ties lexicographic) → top_k → `selected` shape as
`embedding_cosine.py` / `mderank.py`. Unlike MDERank/HCUKE it does **not**
consume the injected `Embedder`; it owns a T5 encoder-decoder.

Two seams keep it importable in the lite env and testable without models:

- **Lazy `transformers` import.** The T5 backend imports `transformers` *inside*
  its constructor, never at module top level — mirroring
  `SentenceTransformerEmbedder`'s documented lazy import. `import
  lattice.adapters` (which the registry triggers) therefore succeeds in the
  no-ml environment, and no model loads at import time.
- **Injectable scoring backend.** The adapter depends on a small callable
  `backend(document: str, candidates: Sequence[str]) -> dict[str, float]`
  returning a per-candidate mean-log-probability. The default backend
  (`T5PromptBackend`) lazy-loads `t5-base`; **tests inject a deterministic fake**
  (no model, no download), exactly as `test_mderank_scorer.py` injects a
  `HashingEmbedder`. All adapter logic (length normalization, position penalty,
  ranking, selection) is unit-tested in the default suite; the real T5 path gets
  a separate `@pytest.mark.ml` + `importorskip("transformers")` test that skips
  cleanly when models/ml are absent.

**Reused unchanged:** `registry`, `config/factory.py`, `harness/sweep.py`,
`harness/runner.py`, the `f1-at-k` document metric, `configs/m2b-f1atk.toml`
(the embedding-cosine incumbent single-point config), and
`scripts/interval_analysis.py`'s paired-delta pattern.

## 4. The faithful PromptRank algorithm

PromptRank is unsupervised prompt-based keyphrase ranking on an encoder-decoder
PLM. For each candidate *c* (drawn here from the noun-chunk extractor):

1. The **encoder** sees the document under the paper's encoder template.
2. The **decoder** is teacher-forced through the paper's decoder template with
   *c* filled into the slot (e.g. *"This book mainly talks about ⟨c⟩"*).
3. **Probability score** = length-normalized mean log-probability of *c*'s
   decoder tokens.
4. **Position penalty** — candidates occurring earlier in the document are
   up-weighted (a deterministic function of *c*'s first occurrence position).
5. Final salience combines (3) and (4); rank desc, top_k selected.

**Exact templates and hyperparameters are pinned at plan-writing time** —
transcribed verbatim from the PromptRank paper and its released reference
implementation, cited inline (source + commit). They are *not* guessed in this
spec. The plan's `promptrank` unit test doubles as the machine-verified fixture:
a fake backend with hand-chosen log-probs yields a known ranking, position
ordering, and selection set, so the adapter's arithmetic is checked against a
worked example before commit (standing "machine-verify plan code" constraint).

**Faithfulness ledger** (documented in the adapter docstring, MDERank-style):
- **Model:** `t5-base` — matches the paper.
- **Candidates:** from the injected `noun-chunk` `Extractor` (paper: a POS-regex
  NP chunker) — the same documented deviation MDERank and HCUKE already carry.
- **Document text:** units joined by `"\n"`; Inspec abstracts (~122 words) fit
  T5's 512-token window, so no truncation handling (mirrors MDERank's MiniLM
  window note).
- **Determinism:** inference-only teacher-forced likelihood — no sampling and no
  free generation — the "local deterministic seq2seq" characterization the
  architecture spec used.

## 5. Components & files

**Create:**
- `src/lattice/adapters/scorer/promptrank.py` — the `@register(Scorer,
  "promptrank")` adapter + the default `T5PromptBackend` (lazy `transformers`
  import). Constructor params: `top_k: int = 15`, the model name (default
  `t5-base`), the pinned hyperparameters, and an optional injectable `backend`.
- `tests/adapters/test_promptrank_scorer.py` — fake-backend unit tests
  (ranking/position/selection logic; the `ScorerContract`) with **no model**,
  plus one `@pytest.mark.ml` real-`t5-base` smoke test guarded by
  `importorskip`.
- `configs/m2-promptrank-sweep.toml` — the M2b `[base.*]` with a 5-entry scorer
  axis (the four incumbents + `promptrank`, all `top_k=15`) → the point table.
- `configs/m2-promptrank-f1atk.toml` — single-point `promptrank` config (M2b
  `[base.*]`, `scorer=promptrank`, `top_k=15`) for the item-level bootstrap CI.
- `docs/results/2026-07-29-promptrank-baseline.md` — the committed results doc
  (see §7).

**Modify:**
- `scripts/fetch_models.py` — add the `t5-base` download (the only sanctioned
  model-download path), guarded so it runs under `--group ml`.

**Reused, not modified:** `configs/m2b-f1atk.toml` serves as the incumbent
single-point config for the paired delta **iff** embedding-cosine is the
strongest included scorer at f1@15 (expected — the M2b amendment notes MDERank
does not beat cosine on Inspec's short abstracts). If a different incumbent wins,
create its single-point config at run time; recorded, not silently defaulted.

**Not committed:** `reports/intervals/**` (gitignored). **Unchanged:** every
file under `src/` except the new adapter.

## 6. Run protocol

All runs seed 0, on the Inspec test split (500 docs), as the manual ML-heavy
regeneration step (mirrors Track 1 Task 11 / Track 2 Task 3). Offline:
`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; prefix each command with
`chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null;` (the macOS
`.pth` re-hide quirk).

1. **Fetch the model:** `uv run --group ml python scripts/fetch_models.py`
   (now also caches `t5-base`).
2. **Point table (sweep):** `python -m lattice.harness --sweep
   configs/m2-promptrank-sweep.toml <out>` → F1/P/R @5/10/15 for all five
   scorers under identical conditions. Identifies the strongest incumbent at
   f1@15.
3. **Runtime gate:** after the first few PromptRank documents, confirm the
   per-document scoring cost is tractable (candidate count × short decoder
   passes, batched per document). If a full 500-doc pass would be
   unreasonably long on the available hardware, **surface it and decide**
   (batch tuning / smaller reporting slice, recorded) before committing the full
   run — the same "gate before the long run" discipline as Track 2's diagnostic
   gate.
4. **Bootstrap CIs:** item-level bootstrap (B=10000, the CLI default) on
   PromptRank: `python -m lattice.harness.stats configs/m2-promptrank-f1atk.toml
   reports/intervals/m2-promptrank --seed 0`.
5. **Paired delta:** run PromptRank and the strongest incumbent bundles at the
   same seed on the same 500 docs and call `paired_delta()` — the exact pattern
   `scripts/interval_analysis.py` uses for M3 exact-label vs nn@0.90. Answers
   "is PromptRank's difference from the best incumbent real or within noise?"

## 7. Deliverable

One committed results doc, `docs/results/2026-07-29-promptrank-baseline.md`,
with:

- **Point table:** F1/P/R @5/10/15 for all five scorers on Inspec test, the new
  PromptRank row dropped alongside the published four.
- **CI + paired-delta section:** PromptRank's f1@k with bootstrap CIs, and the
  paired PromptRank-vs-strongest-incumbent delta with its CI and the fraction of
  paired resamples favoring PromptRank.
- **Honest verdict, either way:** "PromptRank beats the included frontier →
  disclosed, the benchmark's method set is now complete and the strongest method
  is named" *or* "PromptRank is comparable/worse → the exclusion hid nothing."
  Both are valid, reportable outcomes.
- **Faithfulness note:** the pinned templates/hyperparameters, their cited
  source, and the documented deviations — so a reader can judge whether "this is
  PromptRank."
- **Regen commands** for every number, matching the Track 1/2 doc style.

## 8. Constraints (inherited, binding)

- **No new dependencies** — `pyproject` frozen; `transformers` is used
  transitively, only a model is added to the sanctioned download path.
- **No model downloads or network in tests** — adapter logic tested via an
  injected fake backend; the real-model test is `@pytest.mark.ml` +
  `importorskip` and skips cleanly. ML runs happen only in the manual regen step.
- **`import lattice.adapters` must succeed without the ml group** — enforced by
  the lazy `transformers` import.
- `reports/**` gitignored; `docs/results/*.md` **is** committed. No fabricated
  numbers — every table cell traces to an on-disk JSON from a recorded command.
- Deterministic, seed-0, inference-only.
- Work on `main`; never merge/tag/push — the user decides.

## 9. Testing

- **Unit (default suite, no model):** inject a fake backend with fixed
  per-candidate log-probs; assert the resulting salience ordering, the
  position-penalty effect, `top_k` selection, and lexicographic tie-breaking.
  Include the empty-mentions edge case and the `ScorerContract`.
- **ML smoke test:** `@pytest.mark.ml`, `importorskip("transformers")`; construct
  the real `t5-base` adapter on a tiny inline document and assert it returns a
  well-formed, deterministic `ScoredMention` list (skips without ml/models).
- **Config validation:** the two new TOMLs load through the existing
  `ExperimentConfig` validation path (same as every committed config).
- **No new metric logic** → no new metric tests; `f1-at-k` is reused unchanged.

## 10. Risks

- **Faithfulness (primary):** the claim "this is PromptRank" rests on the exact
  templates + hyperparameters. **Mitigated** by pinning them from the paper's
  released reference implementation at plan time, citing source/commit, and the
  worked-example unit fixture. If the reference is unreachable, the exact source
  used is documented in the results doc.
- **Runtime:** t5-base scoring ~40 candidates × 500 docs on CPU. Batched per
  document (one encoder pass, batched decoder passes) it is a tens-of-minutes
  manual step; mitigated by the §6 runtime gate before the full run.
- **Incumbent identity:** the paired delta targets the strongest included
  scorer; if it is not embedding-cosine, a single-point config for the actual
  winner is created at run time (recorded).

## 11. Out of scope (YAGNI)

No changes to the shipped M6 engine or any pipeline/harness source; PromptRank is
never a pipeline default. No re-implementation of MDERank/HCUKE; no new dataset;
no second model family; no hyperparameter search (the paper's defaults are
applied as-is); no encoder/decoder template ablation. This track adds exactly one
scorer and reports it.

## 12. Success criteria

Track 3 succeeds when: the `promptrank` adapter is shipped, the default test
suite is green **without** any model or the ml group, and the ml-guarded
real-model test passes when models are present; the real run reproduces a
PromptRank row with bootstrap CIs and a paired delta vs the strongest incumbent;
and a committed results doc states a clear, honest verdict — PromptRank wins or
it does not — with a faithfulness note and full regen commands.
