# PromptRank Excluded-Method Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-admit PromptRank (Kong et al., ACL 2023) — the one keyphrase scorer M2 deliberately excluded — as a `promptrank` `Scorer` adapter, run it under identical M2 conditions on Inspec, and report where it lands relative to the four included scorers with bootstrap CIs and a paired delta.

**Architecture:** One new `Scorer` adapter with an injectable model backend (the deterministic ranking/position logic is model-free-testable; the default `T5PromptBackend` lazy-loads t5-base). Two configs (5-scorer sweep + single-point) drop into the existing M2 harness; the run reuses Track 1's `f1-at-k` bootstrap and `paired_delta`. No pipeline/harness source changes; PromptRank is a comparison entry only, never a pipeline default.

**Tech Stack:** Python 3.12+, pydantic, `transformers` (already resolved transitively via sentence-transformers — *not* a new dependency) + torch behind the `ml` extra, t5-base fast tokenizer (no sentencepiece).

## Global Constraints

Copied verbatim from the spec (`docs/2026-07-29-promptrank-baseline-design.md` §8):

- **No new dependencies** — `pyproject` frozen; `transformers` is used transitively, only a model is added to the sanctioned download path (`scripts/fetch_models.py`).
- **No model downloads or network in tests** — adapter logic tested via an injected fake backend; the real-model test is `@pytest.mark.ml` + `importorskip` and skips cleanly. ML runs happen only in the manual regen step.
- **`import lattice.adapters` must succeed without the ml group** — enforced by the lazy `transformers` import (import inside the backend constructor, never at module top level).
- `reports/**` gitignored; `docs/results/*.md` **is** committed. No fabricated numbers — every table cell traces to an on-disk JSON from a recorded command.
- Deterministic, seed-0, inference-only.
- PromptRank stays **comparison-only** — never added to the M6 engine defaults or any shipped pipeline.
- Work on `main`; never merge/tag/push — the user decides.

**Faithfulness constants** (pinned verbatim from NKU-HLT/PromptRank, `master` branch, `main.py`/`inference.py`/`data.py`, retrieved 2026-07-29):
`temp_en="Book:"`, `temp_de="This book mainly talks about "`, `model=t5-base`, `length_factor=0.6`, `position_factor=1.2e8`, `max_len=512`, `enable_pos=True`, `enable_filter=False`. Encoder input `Book:"<doc>"`; decoder input `<temp_de><candidate> .`; `prob_score = (Σ log p(candidate tokens)) / (n_candidate_tokens ** 0.6)`; `pos_norm = pos/doc_len + position_factor/doc_len**3`; `final = pos_norm * prob_score`; rank descending.

---

### Task 1: `promptrank` Scorer adapter (injectable T5 backend)

**Files:**
- Create: `src/lattice/adapters/scorer/promptrank.py`
- Modify: `src/lattice/adapters/__init__.py` (add `promptrank` to the scorer import block so `@register` fires)
- Test: `tests/adapters/test_promptrank_scorer.py`

**Interfaces:**
- Consumes: `Scorer` port (`score(mentions: Sequence[Mention], units: Sequence[Unit]) -> list[ScoredMention]`); `Mention`/`ScoredMention`/`Unit` from `lattice.core.types`; `@register(Scorer, "promptrank")`.
- Produces: `PromptRankScorer(top_k: int = 15, model: str = "t5-base", backend: PromptBackend | None = None)`; abstract `PromptBackend.score_candidates(document: str, candidates: Sequence[str]) -> dict[str, float]` (returns each candidate's length-normalized prompt log-likelihood); default `T5PromptBackend`. Registry name `"promptrank"`. Salience convention: `pos_norm * prob_score` (negative; ranked descending, ties lexicographic), matching how `f1-at-k` reads `selected_mentions`.

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/test_promptrank_scorer.py`:

```python
import math

import pytest

from lattice.adapters.scorer.promptrank import PromptBackend, PromptRankScorer
from lattice.ports import Scorer
from lattice.registry.registry import lookup
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class DictBackend(PromptBackend):
    """Test double: returns a preset log-prob per candidate (default -1.0),
    so the adapter's position/ranking logic is exercised without a model."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = -1.0):
        self._scores = scores or {}
        self._default = default

    def score_candidates(self, document, candidates):
        return {c: self._scores.get(c, self._default) for c in candidates}


class TestPromptRankScorer(ScorerContract):
    def make_scorer(self) -> Scorer:
        return PromptRankScorer(backend=DictBackend())

    def test_registered_under_promptrank(self):
        assert lookup(Scorer, "promptrank") is PromptRankScorer

    def test_default_backend_is_lazy_not_built_at_construction(self):
        # No model/transformers import until score() is first called.
        scorer = PromptRankScorer()
        assert scorer._backend is None

    def test_earlier_candidate_wins_when_probability_ties(self):
        scorer = PromptRankScorer(top_k=1, backend=DictBackend({"alpha": -2.0, "beta": -2.0}))
        unit = make_unit(id="d:u0", text="alpha beta gamma alpha")  # doc_len=4
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        scored = {sm.mention.surface: sm for sm in scorer.score(mentions, [unit])}
        # K = 1.2e8 / 4**3 = 1_875_000.0
        # alpha pos=0 -> pos_norm=1_875_000.0     -> salience=-3_750_000.0
        # beta  pos=1 -> pos_norm=1_875_000.25    -> salience=-3_750_000.5
        assert scored["alpha"].salience == pytest.approx(-3_750_000.0)
        assert scored["beta"].salience == pytest.approx(-3_750_000.5)
        assert scored["alpha"].selected and not scored["beta"].selected

    def test_probability_dominates_position(self):
        scorer = PromptRankScorer(top_k=1, backend=DictBackend({"alpha": -10.0, "beta": -2.0}))
        unit = make_unit(id="d:u0", text="alpha beta gamma alpha")  # doc_len=4
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        scored = {sm.mention.surface: sm for sm in scorer.score(mentions, [unit])}
        # alpha: 1_875_000.0  * -10 = -18_750_000.0
        # beta:  1_875_000.25 * -2  =  -3_750_000.5
        assert scored["alpha"].salience == pytest.approx(-18_750_000.0)
        assert scored["beta"].salience == pytest.approx(-3_750_000.5)
        assert scored["beta"].selected and not scored["alpha"].selected

    def test_empty_units_disable_the_position_term(self):
        scorer = PromptRankScorer(backend=DictBackend({"alpha": -2.0}))
        m = make_mention(surface="alpha", unit_id="d:u0", span=(0, 5))
        scored = scorer.score([m], [])  # no units -> doc_len 0
        assert scored[0].salience == pytest.approx(-2.0)


@pytest.mark.ml
def test_real_t5_backend_is_finite_and_deterministic():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    scorer_a = PromptRankScorer(top_k=2)  # default T5PromptBackend, t5-base
    unit = make_unit(
        id="d:u0", text="Graph neural networks learn node embeddings for graph data."
    )
    mentions = [
        make_mention(surface="graph neural networks", unit_id="d:u0", span=(0, 21)),
        make_mention(surface="node embeddings", unit_id="d:u0", span=(28, 43)),
    ]
    a = scorer_a.score(mentions, [unit])
    b = PromptRankScorer(top_k=2).score(mentions, [unit])
    assert {sm.mention.span for sm in a} == {m.span for m in mentions}  # every mention scored
    assert all(math.isfinite(sm.salience) for sm in a)
    assert [round(sm.salience, 4) for sm in a] == [round(sm.salience, 4) for sm in b]  # deterministic
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_promptrank_scorer.py -v -m "not ml"`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer.promptrank'`.

- [ ] **Step 3: Write the adapter**

Create `src/lattice/adapters/scorer/promptrank.py`:

```python
from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.ports import Scorer
from lattice.registry.registry import register

# PromptRank defaults, pinned verbatim from NKU-HLT/PromptRank (master:
# main.py / inference.py / data.py, retrieved 2026-07-29).
TEMP_EN = "Book:"
TEMP_DE = "This book mainly talks about "
LENGTH_FACTOR = 0.6
POSITION_FACTOR = 1.2e8
MAX_LEN = 512


class PromptBackend(ABC):
    """Computes each candidate's length-normalized prompt log-likelihood
    (PromptRank's prob_score). Isolated behind this seam so the position/
    ranking logic is testable without a model — tests inject a fake, exactly
    as the MDERank test injects a fake Embedder."""

    @abstractmethod
    def score_candidates(
        self, document: str, candidates: Sequence[str]
    ) -> dict[str, float]: ...


@register(Scorer, "promptrank")
class PromptRankScorer(Scorer):
    """PromptRank (Kong et al., ACL 2023): rank each candidate by the
    length-normalized log-probability of a T5 decoder generating it from a
    prompt, then apply a position penalty favouring earlier candidates.
    Faithful to NKU-HLT/PromptRank (see module constants).

    PromptRank is admitted here as a benchmark comparison scorer only; it is
    never a shipped-pipeline default, so the "no generative LLM on the critical
    path" property of the delivered engine is preserved (Track 3 spec §1).

    Documented deviations (faithfulness ledger):
    - Candidates come from the pipeline's injected Extractor (paper: a
      StanfordCoreNLP POS-regex NP chunker). Same deviation MDERank/HCUKE carry.
    - Tokenizer: T5 *fast* tokenizer (paper: the slow T5Tokenizer). Equivalent
      for T5; the fast tokenizer avoids a `sentencepiece` dependency (pyproject
      is frozen).
    - `pos`/`doc_len` use whitespace word tokenization of the joined units
      (paper: CoreNLP word indices). A mild tie-breaker; the constant
      position_factor/doc_len**3 term dominates within a document.
    - The model backend is injectable; the default (`T5PromptBackend`) loads
      t5-base lazily so this module imports without the ml group."""

    def __init__(
        self,
        top_k: int = 15,
        model: str = "t5-base",
        backend: "PromptBackend | None" = None,
    ):
        self.top_k = top_k
        self.model = model
        self._backend = backend  # built lazily on first score() if None

    def _get_backend(self) -> PromptBackend:
        if self._backend is None:
            self._backend = T5PromptBackend(self.model)
        return self._backend

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        document = " ".join(unit.text for unit in units)
        words = document.split()
        doc_len = len(words)
        surfaces = sorted({m.surface for m in mentions})
        prob = self._get_backend().score_candidates(document, surfaces)
        positions = _first_word_positions(words, surfaces)
        salience = {
            s: _positioned_score(prob[s], positions[s], doc_len) for s in surfaces
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {s for s, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]


def _positioned_score(prob_score: float, pos: int, doc_len: int) -> float:
    """PromptRank final score (inference.py L124-127): pos_norm * prob_score
    with pos_norm = pos/doc_len + position_factor/doc_len**3. doc_len == 0 (no
    units) disables the position term."""
    if doc_len == 0:
        return prob_score
    pos_norm = pos / doc_len + POSITION_FACTOR / (doc_len**3)
    return pos_norm * prob_score


def _first_word_positions(
    words: Sequence[str], surfaces: Sequence[str]
) -> dict[str, int]:
    """Earliest whitespace-word index at which each surface's first token
    appears (case-insensitive); 0 if not found (candidates come from this
    document, so normally it is found)."""
    lower = [w.lower() for w in words]
    out: dict[str, int] = {}
    for surface in surfaces:
        parts = surface.split()
        head = parts[0].lower() if parts else surface.lower()
        out[surface] = lower.index(head) if head in lower else 0
    return out


class T5PromptBackend(PromptBackend):
    """Default backend: faithful t5-base scoring. torch/transformers are
    imported lazily inside __init__ so this module is importable without the
    ml dependency group (mirrors SentenceTransformerEmbedder)."""

    def __init__(self, model: str = "t5-base", device: str = "cpu"):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model, model_max_length=MAX_LEN)
        self._model = T5ForConditionalGeneration.from_pretrained(model).to(device).eval()
        self._device = device
        # Decoder tokens belonging to the prompt prefix (inference.py L71):
        # tokens(temp_de) minus the reference's fixed -3 offset.
        self._template_len = (
            self._tok(TEMP_DE, return_tensors="pt")["input_ids"].shape[1] - 3
        )

    def score_candidates(
        self, document: str, candidates: Sequence[str]
    ) -> dict[str, float]:
        torch = self._torch
        candidates = list(candidates)
        if not candidates:
            return {}
        # Truncate to the first MAX_LEN words, wrap in the encoder template
        # (data.py L401, L424): Book:"<doc>".
        doc = " ".join(document.split()[:MAX_LEN])
        enc = self._tok(
            f'{TEMP_EN}"{doc}"', max_length=MAX_LEN, truncation=True, return_tensors="pt"
        ).to(self._device)
        # Decoder inputs: temp_de + candidate + " ." (data.py L323-326).
        de_texts = [f"{TEMP_DE}{c.lower()} ." for c in candidates]
        de = self._tok(
            de_texts, max_length=30, padding="max_length", truncation=True,
            return_tensors="pt",
        )["input_ids"]
        de[:, 0] = 0  # force decoder-start token (data.py L325)
        de = de.to(self._device)
        eos = self._tok.eos_token_id
        de_lens = [int((row == eos).nonzero()[0].item()) - 2 for row in de]  # data.py L326
        n = de.shape[0]
        scores: dict[str, float] = {}
        with torch.no_grad():
            logits = self._model(
                input_ids=enc["input_ids"].repeat(n, 1),
                attention_mask=enc["attention_mask"].repeat(n, 1),
                decoder_input_ids=de,
            ).logits
            logp = torch.log_softmax(logits, dim=-1)
            for j, cand in enumerate(candidates):
                de_len = de_lens[j]
                total = 0.0
                for i in range(self._template_len, de_len):  # inference.py L95-102
                    total += float(logp[j, i, int(de[j, i + 1])])
                count = de_len - self._template_len
                scores[cand] = total / (count**LENGTH_FACTOR) if count > 0 else total
        return scores
```

- [ ] **Step 4: Register the adapter**

In `src/lattice/adapters/__init__.py`, add `promptrank` to the scorer import block (keep alphabetical order):

```python
from lattice.adapters.scorer import (  # noqa: F401
    embedding_cosine,
    frequency,
    hcuke,
    mderank,
    passthrough,
    promptrank,
)
```

- [ ] **Step 5: Run the non-ml tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_promptrank_scorer.py -v -m "not ml"`
Expected: PASS — the 3 `ScorerContract` tests + `test_registered_under_promptrank`, `test_default_backend_is_lazy_not_built_at_construction`, `test_earlier_candidate_wins_when_probability_ties`, `test_probability_dominates_position`, `test_empty_units_disable_the_position_term`. The `@pytest.mark.ml` test is deselected.

- [ ] **Step 6: Confirm the full default suite stays green (registration didn't break imports)**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q -m "not ml"`
Expected: PASS — all previously-passing tests plus the new ones; no collection/import errors.

- [ ] **Step 7: Commit**

```bash
git add src/lattice/adapters/scorer/promptrank.py src/lattice/adapters/__init__.py tests/adapters/test_promptrank_scorer.py
git commit -m "feat: PromptRank scorer adapter with injectable T5 backend (track 3)"
```

---

### Task 2: M2 PromptRank configs + t5-base fetch

**Files:**
- Create: `configs/m2-promptrank-sweep.toml`, `configs/m2-promptrank-f1atk.toml`
- Modify: `scripts/fetch_models.py`
- Test: `tests/harness/test_m2_promptrank_configs.py`

**Interfaces:**
- Consumes: `SweepConfig` + `expand` from `lattice.harness.sweep`; `ExperimentConfig` from `lattice.harness.runner`; `load_config` from `lattice.config.loader`; the `promptrank` registry name from Task 1.
- Produces: two loadable configs on the Inspec test split; `fetch_models.py` additionally caches t5-base.

- [ ] **Step 1: Write the failing config-validation tests**

Create `tests/harness/test_m2_promptrank_configs.py`:

```python
from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig
from lattice.harness.sweep import SweepConfig, expand


def test_promptrank_sweep_expands_to_five_scorers():
    sweep = load_config("configs/m2-promptrank-sweep.toml", model=SweepConfig)
    configs = expand(sweep)
    assert [c.scorer.name for c in configs] == [
        "frequency", "embedding-cosine", "mderank", "hcuke", "promptrank",
    ]
    for c in configs:
        assert c.scorer.params["top_k"] == 15
        assert c.dataset.params["split"] == "test"
        assert c.dataset.name == "inspec"


def test_promptrank_single_config_loads():
    cfg = load_config("configs/m2-promptrank-f1atk.toml", model=ExperimentConfig)
    assert cfg.scorer.name == "promptrank"
    assert cfg.scorer.params["top_k"] == 15
    assert cfg.dataset.name == "inspec" and cfg.dataset.params["split"] == "test"
    assert any(dm.name == "f1-at-k" for dm in cfg.document_metrics)
```

- [ ] **Step 2: Run to verify failure**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m2_promptrank_configs.py -v`
Expected: FAIL — `FileNotFoundError`/load error (configs do not exist yet).

- [ ] **Step 3: Create the sweep config**

Create `configs/m2-promptrank-sweep.toml`:

```toml
# M2 excluded-method baseline (credibility Track 3): the four M2b scorers plus
# PromptRank under identical conditions — sentence units, noun-chunk candidates,
# MiniLM embeddings — on the Inspec test split. Same [base.*] as
# configs/m2b-sweep.toml; PromptRank added as a fifth scorer-axis entry.
# PromptRank ignores the shared embedder (it owns a T5); the embedder stays for
# base-config uniformity.

[base.segmenter]
name = "sentence"

[base.extractor]
name = "noun-chunk"

[base.resolver]
name = "exact-label"

[base.relation_inducer]
name = "co-occurrence"

[base.graph_integrator]
name = "in-memory"

[base.embedder]
name = "sentence-transformer"

[base.concept_store]
name = "in-memory"

[base.run]
on_error = "fail"
seed = 0

[base.dataset]
name = "inspec"
[base.dataset.params]
split = "test"

[base.scorer]
name = "embedding-cosine"

[[base.document_metrics]]
name = "f1-at-k"

# Invariant: scorer top_k must be >= max ks of f1-at-k (selected mentions cap the ranking depth).
[axes]
scorer = [
  { name = "frequency", params = { top_k = 15 } },
  { name = "embedding-cosine", params = { top_k = 15 } },
  { name = "mderank", params = { top_k = 15 } },
  { name = "hcuke", params = { top_k = 15 } },
  { name = "promptrank", params = { top_k = 15 } },
]
```

- [ ] **Step 4: Create the single-point config**

Create `configs/m2-promptrank-f1atk.toml`:

```toml
# PromptRank single-config (credibility Track 3): collapsed from
# configs/m2-promptrank-sweep.toml's [base.*], scorer=promptrank at top_k=15
# (f1-at-k invariant: scorer top_k >= max(ks)=15), on the Inspec test split.
# For item-level bootstrap CI via `python -m lattice.harness.stats`.

[segmenter]
name = "sentence"

[extractor]
name = "noun-chunk"

[scorer]
name = "promptrank"
[scorer.params]
top_k = 15

[resolver]
name = "exact-label"

[relation_inducer]
name = "co-occurrence"

[graph_integrator]
name = "in-memory"

[embedder]
name = "sentence-transformer"

[concept_store]
name = "in-memory"

[run]
on_error = "fail"
seed = 0

[dataset]
name = "inspec"
[dataset.params]
split = "test"

[[document_metrics]]
name = "f1-at-k"
```

- [ ] **Step 5: Run the config tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m2_promptrank_configs.py -v`
Expected: PASS — both tests.

- [ ] **Step 6: Add t5-base to the sanctioned download path**

In `scripts/fetch_models.py`, append to `main()` (after the sentence-transformer block, before the function returns):

```python
    # PromptRank baseline (credibility Track 3): cache t5-base via the fast
    # tokenizer (no sentencepiece dependency; pyproject frozen).
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    AutoTokenizer.from_pretrained("t5-base")
    T5ForConditionalGeneration.from_pretrained("t5-base")
    print("t5-base ready (PromptRank)")
```

- [ ] **Step 7: Commit**

```bash
git add configs/m2-promptrank-sweep.toml configs/m2-promptrank-f1atk.toml scripts/fetch_models.py tests/harness/test_m2_promptrank_configs.py
git commit -m "feat: M2 PromptRank configs + t5-base fetch (track 3)"
```

---

### Task 3: (main session) Real run, paired-delta driver, results doc

> **Executed by the main session, not delegated** — this is the manual, ML-heavy, non-committed-data task (mirrors Track 1 Task 11 / Track 2 Task 3). It requires the `ml` extra + cached models. It produces gitignored reports under `reports/` and the **committed** results doc + the committed driver script. No fabricated numbers: every table cell is populated from an on-disk JSON produced by a recorded command.

**Files:**
- Create: `scripts/promptrank_analysis.py` (committed paired-delta driver)
- Create: `docs/results/2026-07-29-promptrank-baseline.md` (committed results doc)
- Produces (gitignored): `reports/m2-promptrank-sweep/`, `reports/intervals/m2-promptrank/`, `reports/intervals/promptrank/promptrank-paired-delta.json`

**Interfaces:**
- Consumes: `run_experiment_detailed` (`lattice.harness.runner`), `bootstrap` (`lattice.harness.stats.resample`), `paired_delta`/`DeltaResult` (`lattice.harness.stats.intervals`); `report.metrics["f1-at-k"]["f1@{k}"]` point estimates and `bundles["f1-at-k"]` resample bundle.

- [ ] **Step 1: Write the paired-delta driver**

Create `scripts/promptrank_analysis.py`:

```python
"""PromptRank-vs-incumbent paired-delta bootstrap on Inspec f1@k, for
docs/results/2026-07-29-promptrank-baseline.md (credibility Track 3).

Pure orchestration over run_experiment_detailed + bootstrap + paired_delta —
the same primitives scripts/interval_analysis.py uses for M3's paired delta;
no new statistical algorithms. Both configs run on the same 500-document
Inspec test split, so the same seed draws identical document indices at each
iteration (paired by construction).

    uv run --no-sync python scripts/promptrank_analysis.py \
        [out_dir] --incumbent configs/m2b-f1atk.toml --incumbent-label embedding-cosine
"""

import argparse
import json
from pathlib import Path

from lattice.config.loader import load_config
from lattice.harness.runner import ExperimentConfig, run_experiment_detailed
from lattice.harness.stats.intervals import DeltaResult, paired_delta
from lattice.harness.stats.resample import bootstrap

PROMPTRANK_CONFIG = "configs/m2-promptrank-f1atk.toml"
ITEM_SAMPLES = 10000  # matches the CLI's item-level default (Track 1)
SEED = 0
LEVEL = 0.95
KS = ["f1@5", "f1@10", "f1@15"]


def _f1atk(path: str):
    cfg = load_config(path, model=ExperimentConfig)
    report, bundles = run_experiment_detailed(cfg)
    return report.metrics["f1-at-k"], bundles["f1-at-k"]


def paired(incumbent_path: str, incumbent_label: str) -> dict:
    pr_est, pr_bundle = _f1atk(PROMPTRANK_CONFIG)
    in_est, in_bundle = _f1atk(incumbent_path)
    pr_res = bootstrap(pr_bundle, samples=ITEM_SAMPLES, seed=SEED)
    in_res = bootstrap(in_bundle, samples=ITEM_SAMPLES, seed=SEED)
    rows = []
    for k in KS:
        d: DeltaResult = paired_delta(pr_res[k], in_res[k], pr_est[k], in_est[k], level=LEVEL)
        rows.append({
            "metric": k,
            "promptrank": pr_est[k],
            "incumbent": in_est[k],
            "delta_estimate": d.estimate,
            "ci_lo": d.lo,
            "ci_hi": d.hi,
            "prob_positive": d.prob_positive,
        })
    return {"incumbent_label": incumbent_label, "samples": ITEM_SAMPLES, "seed": SEED, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(prog="promptrank_analysis")
    p.add_argument("out_dir", nargs="?", default="reports/intervals/promptrank")
    p.add_argument("--incumbent", default="configs/m2b-f1atk.toml")
    p.add_argument("--incumbent-label", default="embedding-cosine")
    args = p.parse_args()
    result = paired(args.incumbent, args.incumbent_label)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "promptrank-paired-delta.json").write_text(json.dumps(result, indent=2))
    print(f"=== PromptRank - {args.incumbent_label} paired delta (Inspec f1@k) ===")
    for r in result["rows"]:
        print(
            f"{r['metric']}: promptrank={r['promptrank']:.4f} "
            f"{args.incumbent_label}={r['incumbent']:.4f} "
            f"delta={r['delta_estimate']:+.4f} "
            f"95% CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
            f"prob_positive={r['prob_positive']:.4f}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Fetch the model**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; export SSL_CERT_FILE=$(uv run --no-sync python -c "import certifi; print(certifi.where())"); uv run --group ml python scripts/fetch_models.py`
Expected: prints `t5-base ready (PromptRank)` (and the sentence-transformer line). This is the only step that touches the network.

- [ ] **Step 3: Verify the real backend end-to-end (ml smoke test)**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; uv run --no-sync pytest tests/adapters/test_promptrank_scorer.py::test_real_t5_backend_is_finite_and_deterministic -v -m ml`
Expected: PASS — finite, deterministic salience from the real t5-base backend.

- [ ] **Step 4: Point-estimate sweep (all 5 scorers)**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; uv run --no-sync python -m lattice.harness --sweep configs/m2-promptrank-sweep.toml reports/m2-promptrank-sweep`
Expected: a sweep table with F1/P/R @5/10/15 for `frequency`/`embedding-cosine`/`mderank`/`hcuke`/`promptrank`; JSON written under `reports/m2-promptrank-sweep/`. **Runtime gate:** if PromptRank's pass is impractically slow on the available hardware, surface it and decide (batch tuning / recorded reporting slice) before continuing — do not silently truncate. **Record which incumbent is strongest at f1@15** for Step 6.

- [ ] **Step 5: PromptRank marginal bootstrap CIs**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; uv run --no-sync python -m lattice.harness.stats configs/m2-promptrank-f1atk.toml reports/intervals/m2-promptrank --seed 0`
Expected: an interval report JSON with PromptRank `f1@5/10/15` point estimates + percentile/BCa CIs (B=10000).

- [ ] **Step 6: Paired delta vs the strongest incumbent**

Run (default incumbent = embedding-cosine via the committed `configs/m2b-f1atk.toml`; if Step 4 named a different strongest incumbent, first create its single-point config by collapsing `configs/m2-promptrank-sweep.toml`'s `[base.*]` with that scorer, and pass `--incumbent <that config> --incumbent-label <name>`):

`chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; uv run --no-sync python scripts/promptrank_analysis.py reports/intervals/promptrank --incumbent configs/m2b-f1atk.toml --incumbent-label embedding-cosine`
Expected: prints the paired delta per f1@k and writes `reports/intervals/promptrank/promptrank-paired-delta.json`.

- [ ] **Step 7: Write the results doc from the JSONs**

Create `docs/results/2026-07-29-promptrank-baseline.md`. Populate **every** number from the on-disk JSONs produced in Steps 4–6 (no estimates). Structure:

```markdown
# PromptRank Excluded-Method Baseline — does re-admitting the excluded generative scorer change the M2 story?

Track 3 deliverable. Adds PromptRank (Kong et al., ACL 2023) — the one keyphrase
scorer M2 deliberately excluded — as a `promptrank` Scorer and runs it under
identical M2b conditions on the Inspec test split (500 docs, seed 0). PromptRank
is a comparison entry only; it is never wired into the shipped engine. Every
number below was produced by the recorded command (§5) against this repository's
code and the cached t5-base. Runs executed 2026-07-29, seed 0.

## 1. Headline verdict
[One paragraph: does PromptRank beat / match / trail the four included scorers at
f1@15? State it plainly, with the paired-delta outcome (CI excludes zero or not).
Either outcome is a valid result: a win → the benchmark's strongest method is now
named and disclosed; a non-win → the exclusion hid nothing.]

## 2. Point table — five scorers on Inspec test (top_k=15)
[From reports/m2-promptrank-sweep/. P/R/F1 @5/10/15 for frequency,
embedding-cosine, mderank, hcuke, promptrank. Mark the strongest at f1@15.]

## 3. PromptRank CIs + paired delta vs the strongest incumbent
[From reports/intervals/m2-promptrank/ (marginal CIs) and
reports/intervals/promptrank/promptrank-paired-delta.json (paired delta:
estimate, 95% CI, prob_positive per f1@k). State whether the delta CI excludes 0.]

## 4. Interpretation + faithfulness note
[What the result means for the "we compared the frontier" claim. Then the
faithfulness ledger: pinned constants (temp_en/temp_de/length_factor=0.6/
position_factor=1.2e8/t5-base), the cited source (NKU-HLT/PromptRank master,
commit from §5), and the documented deviations (extractor candidates; fast
tokenizer; whitespace position tokenization).]

## 5. Regeneration
[The exact commands from Steps 2, 4, 5, 6, with the HF_HUB_OFFLINE + chflags
prefixes. Record the PromptRank reference commit:
`git ls-remote https://github.com/NKU-HLT/PromptRank master`.]
```

- [ ] **Step 8: Commit the driver + results doc**

```bash
git add scripts/promptrank_analysis.py docs/results/2026-07-29-promptrank-baseline.md
git commit -m "docs: PromptRank excluded-method baseline results (track 3)"
```

---

## Self-Review

**1. Spec coverage** (against `docs/2026-07-29-promptrank-baseline-design.md`):
- §2 faithful t5-base / no-dep / comparison-only → Task 1 adapter (T5PromptBackend, lazy import, docstring) + Global Constraints.
- §3 injectable backend + lazy import → Task 1 (`PromptBackend` seam, `_backend is None` test, `@pytest.mark.ml` real test).
- §4 algorithm + faithfulness ledger → Task 1 constants, `_positioned_score`, `T5PromptBackend`, docstring deviations.
- §5 files → Task 1 (adapter+init+test), Task 2 (2 configs + fetch_models), Task 3 (driver + results doc).
- §6 run protocol (fetch → sweep → runtime gate → CIs → paired delta) → Task 3 Steps 2–6.
- §7 deliverable (point table + CI/paired-delta + honest verdict + faithfulness note + regen) → Task 3 Step 7 skeleton.
- §8 constraints → Global Constraints block.
- §9 testing (fake-backend unit + ml smoke + config validation) → Task 1 tests + Task 2 test.
- §10 risks (faithfulness pinned from source; runtime gate; incumbent identity) → Global Constraints faithfulness constants; Task 3 Step 4 gate + Step 6 incumbent handling.
No gaps.

**2. Placeholder scan:** Task 3's results-doc skeleton uses bracketed populate-from-JSON instructions — this is a data-generation task whose numbers cannot exist before the run (same pattern as Track 1 Task 11 / Track 2 Task 3), not a code placeholder. All code steps carry complete code. No "TBD"/"handle errors"/"similar to".

**3. Type consistency:** `PromptRankScorer(top_k, model, backend)`, `PromptBackend.score_candidates(document, candidates) -> dict[str, float]`, and registry name `"promptrank"` are identical across Task 1 (impl + tests), Task 2 (configs reference the name), and Task 3 (configs). `report.metrics["f1-at-k"]["f1@{k}"]` + `bundles["f1-at-k"]` + `bootstrap(...)["f1@{k}"]` + `paired_delta(a, b, est_a, est_b, level=)` match the verified signatures in `runner.py`, `resample.py`, and `intervals.py`. Salience convention (negative, ranked descending, ties lexicographic) matches how `f1-at-k` reads `selected_mentions`.
