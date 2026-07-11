# M2b — Frontier Scorers (MDERank + HCUKE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the MDERank and HCUKE scorer adapters and produce one sweep table comparing all four scorers (frequency, embedding-cosine, mderank, hcuke) on the Inspec test split.

**Architecture:** Two new `Scorer` adapters consuming the existing `Embedder` port, plus a stdlib
sentence segmenter (HCUKE's sentence level) and a pure masked-document helper (MDERank). No new
dependencies of any kind. Spec: `docs/2026-07-08-m2-extraction-salience-design.md` §6.6/§6.7/§11
(as amended 2026-07-10).

**Tech Stack:** Python stdlib only for all new code. The ml group (spaCy/sentence-transformers)
is exercised only by ml-marked e2e tests and the final sweep.

## Global Constraints

- `pyproject.toml` is **FROZEN**. M2b adds zero dependencies. Do not edit it for any reason.
- macOS quirk: `.pth` files go UF_HIDDEN after any implicit sync. Every test/run command must be:
  `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync <cmd>`
- The lean suite must stay green without the ml group; ml tests use
  `pytest.importorskip` + `@pytest.mark.ml` + the `except OSError: pytest.skip(...)` guard.
- No model or dataset downloads inside adapters or tests, ever. `data/` and `reports/` are
  gitignored and never committed. `docs/papers/` is gitignored (copyrighted PDF) — never commit it.
- `uv run --no-sync ruff check` must pass before every commit (line length 100, rules E/F/I/UP).
- Follow existing idioms exactly: `@register(Port, "name")`, sorted-surface determinism,
  `(-salience, surface)` tie-breaking, contract-suite subclassing in adapter tests.
- Do not redesign anything. The plan text is authoritative; if reality contradicts it, STOP and
  report to the orchestrator instead of improvising.

## Paper fidelity (context for reviewers)

- **MDERank** = Zhang et al., Findings of ACL 2022 (arXiv 2110.06651). Verified against the PDF.
- **HCUKE** = Xu et al., Knowledge-Based Systems 304 (2024) 112511. Verified against the paper's
  Algorithm 1 (the PDF is at `docs/papers/hcuke-knosys-2024.pdf`, gitignored).
- Sanctioned deviations (documented in adapter docstrings): candidates come from the injected
  Extractor (papers: POS-regex); embeddings come from the injected Embedder as whole-string
  MiniLM vectors (papers: BERT token vectors + max-pooling); HCUKE word positions use whitespace
  tokens (paper: CoreNLP tokens); sentence splitting is a stdlib regex (paper: CoreNLP).
- Known paper-internal inconsistency in HCUKE: Eq. (5) prose applies the sentence position
  weight twice (it already sits inside Eq. (4)); Algorithm 1 and the worked example in §3.3
  apply it once. **We implement Algorithm 1** (single `W(s)` factor).
- Recommended implementer models: T1/T2 haiku; T3/T4/T5 sonnet. Reviewers sonnet; final review opus.

---

### Task 1: Sentence segmenter

**Files:**
- Create: `src/lattice/adapters/segmenter/sentence.py`
- Modify: `src/lattice/adapters/__init__.py` (segmenter import line)
- Test: `tests/adapters/test_sentence_segmenter.py`

**Interfaces:**
- Consumes: `Segmenter` port (`segment(document: Document) -> list[Unit]`), `Document`/`Unit`
  from `lattice.core.types`, `register` from `lattice.registry.registry`.
- Produces: registered segmenter `"sentence"` yielding `Unit(kind="sentence")` with ids
  `f"{document.id}:u{i}"` and `order` from 0. Task 4's HCUKE runs and Task 5's configs use it.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_sentence_segmenter.py`:

```python
from lattice.adapters.segmenter.sentence import SentenceSegmenter
from tests.contracts.segmenter_contract import SegmenterContract
from tests.helpers import make_document


class TestSentenceSegmenter(SegmenterContract):
    def make_segmenter(self) -> SentenceSegmenter:
        return SentenceSegmenter()

    def test_splits_on_terminal_punctuation(self):
        doc = make_document(text="Alpha is here. Beta follows! Is gamma third? Delta ends")
        assert [u.text for u in self.make_segmenter().segment(doc)] == [
            "Alpha is here.",
            "Beta follows!",
            "Is gamma third?",
            "Delta ends",
        ]

    def test_units_are_kind_sentence(self):
        doc = make_document(text="One. Two.")
        units = self.make_segmenter().segment(doc)
        assert units and all(u.kind == "sentence" for u in units)

    def test_no_split_without_following_whitespace(self):
        doc = make_document(text="Version 2.5 of the system works.")
        assert len(self.make_segmenter().segment(doc)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_sentence_segmenter.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.segmenter.sentence'`

- [ ] **Step 3: Implement the adapter**

`src/lattice/adapters/segmenter/sentence.py`:

```python
import re

from lattice.core.types import Document, Unit
from lattice.ports import Segmenter
from lattice.registry.registry import register

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@register(Segmenter, "sentence")
class SentenceSegmenter(Segmenter):
    """Splits document text into sentences on terminal punctuation followed
    by whitespace. Deterministic stdlib rule providing HCUKE's sentence level
    (M2 spec §6.7); adequate for benchmark abstracts. Documented deviation
    from the paper's CoreNLP sentence tokenizer: no abbreviation handling."""

    def segment(self, document: Document) -> list[Unit]:
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(document.text)]
        sentences = [s for s in sentences if s]
        return [
            Unit(
                id=f"{document.id}:u{i}",
                document_id=document.id,
                text=sentence,
                order=i,
                kind="sentence",
            )
            for i, sentence in enumerate(sentences)
        ]
```

In `src/lattice/adapters/__init__.py`, change the segmenter import line to:

```python
from lattice.adapters.segmenter import block, sentence  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_sentence_segmenter.py -q`
Expected: 7 passed (4 contract + 3 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green (lean run: ml tests skip)

```bash
git add src/lattice/adapters/segmenter/sentence.py src/lattice/adapters/__init__.py tests/adapters/test_sentence_segmenter.py
git commit -m "feat: add stdlib sentence segmenter (HCUKE sentence level)"
```

---

### Task 2: Masked-document helper (pure)

**Files:**
- Create: `src/lattice/adapters/scorer/masking.py`
- Test: `tests/adapters/test_masking.py`

**Interfaces:**
- Consumes: `Mention`/`Unit` from `lattice.core.types`. Nothing else.
- Produces: `mask_document(units: Sequence[Unit], mentions: Sequence[Mention], mask_token: str = "[MASK]") -> str`
  — the document text (units joined by `"\n"`) with every given mention's span replaced.
  Task 3's MDERank scorer calls it with all mentions of one surface.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_masking.py`:

```python
from lattice.adapters.scorer.masking import mask_document
from tests.helpers import make_mention, make_unit


def test_masks_single_occurrence_one_mask_token_per_word():
    unit = make_unit(id="d:u0", text="deep learning wins")
    mention = make_mention(surface="deep learning", unit_id="d:u0", span=(0, 13))
    assert mask_document([unit], [mention]) == "[MASK] [MASK] wins"


def test_masks_all_occurrences_across_units():
    u0 = make_unit(id="d:u0", text="graphs model graphs")
    u1 = make_unit(id="d:u1", document_id="d", text="we like graphs")
    mentions = [
        make_mention(surface="graphs", unit_id="d:u0", span=(0, 6)),
        make_mention(surface="graphs", unit_id="d:u0", span=(13, 19)),
        make_mention(surface="graphs", unit_id="d:u1", span=(8, 14)),
    ]
    assert mask_document([u0, u1], mentions) == "[MASK] model [MASK]\nwe like [MASK]"


def test_mask_token_count_matches_surface_word_count():
    unit = make_unit(id="d:u0", text="convolutional neural network layers")
    mention = make_mention(
        surface="convolutional neural network", unit_id="d:u0", span=(0, 28)
    )
    assert mask_document([unit], [mention]) == "[MASK] [MASK] [MASK] layers"


def test_no_mentions_reproduces_document_text():
    u0 = make_unit(id="d:u0", text="alpha")
    u1 = make_unit(id="d:u1", document_id="d", text="beta")
    assert mask_document([u0, u1], []) == "alpha\nbeta"


def test_custom_mask_token():
    unit = make_unit(id="d:u0", text="alpha beta")
    mention = make_mention(surface="alpha", unit_id="d:u0", span=(0, 5))
    assert mask_document([unit], [mention], mask_token="<mask>") == "<mask> beta"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_masking.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer.masking'`

- [ ] **Step 3: Implement the helper**

`src/lattice/adapters/scorer/masking.py`:

```python
"""Masked-document construction for MDERank (M2 spec §6.6). Pure stdlib."""

from collections.abc import Sequence

from lattice.core.types import Mention, Unit


def mask_document(
    units: Sequence[Unit], mentions: Sequence[Mention], mask_token: str = "[MASK]"
) -> str:
    """Rebuild the document text (units joined by "\\n") with every given
    mention's span replaced by mask tokens — one per whitespace token of the
    mention's surface, preserving sequence length per MDERank §3 ("the number
    of MASK used for masking is as same as the number of tokens"). Spans are
    character offsets into their unit's text and must not overlap."""
    spans_by_unit: dict[str, list[tuple[int, int, str]]] = {}
    for m in mentions:
        replacement = " ".join([mask_token] * max(len(m.surface.split()), 1))
        spans_by_unit.setdefault(m.unit_id, []).append(
            (m.span[0], m.span[1], replacement)
        )
    masked_units = []
    for unit in units:
        text = unit.text
        for start, end, replacement in sorted(spans_by_unit.get(unit.id, []), reverse=True):
            text = text[:start] + replacement + text[end:]
        masked_units.append(text)
    return "\n".join(masked_units)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_masking.py -q`
Expected: 5 passed

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/scorer/masking.py tests/adapters/test_masking.py
git commit -m "feat: add pure masked-document helper for MDERank"
```

---

### Task 3: MDERank scorer

**Files:**
- Create: `src/lattice/adapters/scorer/mderank.py`
- Modify: `src/lattice/adapters/__init__.py` (scorer import line)
- Test: `tests/adapters/test_mderank_scorer.py`

**Interfaces:**
- Consumes: `mask_document(units, mentions, mask_token) -> str` from
  `lattice.adapters.scorer.masking` (Task 2); `Embedder` port
  (`embed(texts: Sequence[str]) -> list[tuple[float, ...]]`, empty string → zero vector);
  `cosine(a, b) -> float` from `lattice.core.vectors` (zero vector → 0.0).
- Produces: registered scorer `"mderank"`, constructor
  `(embedder: Embedder, top_k: int = 10, mask_token: str = "[MASK]")`. Task 5's configs use it.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_mderank_scorer.py`:

```python
from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.mderank import MDERankScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class BatchRecordingEmbedder(Embedder):
    """Test double: hashing embedder that records each embed() batch."""

    def __init__(self):
        self.inner = HashingEmbedder(dim=16)
        self.batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts):
        batch = list(texts)
        self.batches.append(batch)
        return self.inner.embed(batch)


class TestMDERankScorer(ScorerContract):
    def make_scorer(self) -> MDERankScorer:
        return MDERankScorer(embedder=HashingEmbedder(dim=16))

    def test_one_batch_of_document_plus_masked_variants(self):
        embedder = BatchRecordingEmbedder()
        scorer = MDERankScorer(embedder=embedder)
        unit = make_unit(id="d:u0", text="vector store holds a vector")
        mentions = [
            make_mention(surface="vector", unit_id="d:u0", span=(0, 6)),
            make_mention(surface="store", unit_id="d:u0", span=(7, 12)),
            make_mention(surface="vector", unit_id="d:u0", span=(21, 27)),
        ]
        scorer.score(mentions, [unit])
        assert len(embedder.batches) == 1  # single embed call per score()
        batch = embedder.batches[0]
        # [document, masked-per-unique-surface in sorted order: store, vector]
        assert batch[0] == "vector store holds a vector"
        assert batch[1] == "vector [MASK] holds a vector"
        assert batch[2] == "[MASK] store holds a [MASK]"

    def test_masking_a_central_candidate_scores_highest(self):
        # Masking the surface that constitutes most of the document moves the
        # document embedding far more than masking a peripheral one.
        scorer = self.make_scorer()
        unit = make_unit(id="d:u0", text="graph theory graph theory graph theory zebra")
        mentions = [
            make_mention(surface="graph theory", unit_id="d:u0", span=(0, 12)),
            make_mention(surface="graph theory", unit_id="d:u0", span=(13, 25)),
            make_mention(surface="graph theory", unit_id="d:u0", span=(26, 38)),
            make_mention(surface="zebra", unit_id="d:u0", span=(39, 44)),
        ]
        scored = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert scored["graph theory"] > scored["zebra"]

    def test_empty_units_yields_genuine_tie_broken_lexicographically(self):
        # With no units every masked variant equals the empty document; the
        # embedder maps "" to the zero vector, cosine(0, 0) is 0.0, so every
        # salience is exactly 1.0 — a genuine tie.
        scorer = MDERankScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        mentions = [
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        ]
        scored = scorer.score(mentions, [])
        saliences = {sm.mention.surface: sm.salience for sm in scored}
        assert saliences["alpha"] == saliences["beta"] == 1.0
        assert {sm.mention.surface for sm in scored if sm.selected} == {"alpha"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_mderank_scorer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer.mderank'`

- [ ] **Step 3: Implement the adapter**

`src/lattice/adapters/scorer/mderank.py`:

```python
from collections.abc import Sequence

from lattice.adapters.scorer.masking import mask_document
from lattice.core.types import Mention, ScoredMention, Unit
from lattice.core.vectors import cosine
from lattice.ports import Embedder, Scorer
from lattice.registry.registry import register


@register(Scorer, "mderank")
class MDERankScorer(Scorer):
    """MDERank (Zhang et al., Findings of ACL 2022; arXiv 2110.06651): mask
    all occurrences of a candidate and re-embed; candidates whose absence
    moves the document embedding most are most salient. The paper ranks by
    increasing cos(E(doc), E(masked)); salience = 1 - cos(...) is the
    equivalent decreasing form (M2 spec §6.6). One [MASK] per surface word
    preserves sequence length (paper §3).

    Documented deviations: candidates come from the pipeline's injected
    Extractor (paper: POS regex); embeddings from the injected Embedder
    (paper: BERT last layer + max-pooling); "[MASK]" is a literal placeholder
    for the MiniLM family. Inspec abstracts (~122 words) fit MiniLM's
    256-token window, so no truncation handling (spec §6.6)."""

    def __init__(self, embedder: Embedder, top_k: int = 10, mask_token: str = "[MASK]"):
        self.embedder = embedder
        self.top_k = top_k
        self.mask_token = mask_token

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        document_text = "\n".join(unit.text for unit in units)
        surfaces = sorted({m.surface for m in mentions})
        masked_documents = [
            mask_document(
                units, [m for m in mentions if m.surface == surface], self.mask_token
            )
            for surface in surfaces
        ]
        document_vector, *masked_vectors = self.embedder.embed(
            [document_text, *masked_documents]
        )
        salience = {
            surface: 1.0 - cosine(document_vector, masked_vector)
            for surface, masked_vector in zip(surfaces, masked_vectors)
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]
```

In `src/lattice/adapters/__init__.py`, change the scorer import line to:

```python
from lattice.adapters.scorer import embedding_cosine, frequency, mderank  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_mderank_scorer.py -q`
Expected: 6 passed (3 contract + 3 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/scorer/mderank.py src/lattice/adapters/__init__.py tests/adapters/test_mderank_scorer.py
git commit -m "feat: add MDERank scorer adapter"
```

---

### Task 4: HCUKE scorer

**Files:**
- Create: `src/lattice/adapters/scorer/hcuke.py`
- Modify: `src/lattice/adapters/__init__.py` (scorer import line)
- Test: `tests/adapters/test_hcuke_scorer.py`

**Interfaces:**
- Consumes: `Embedder` port, `cosine` from `lattice.core.vectors`, `Mention.span`/`unit_id`,
  `Unit.order`/`text`. Pairs naturally with the `"sentence"` segmenter (Task 1) — units are
  the sentence level — but must work with any segmenter.
- Produces: registered scorer `"hcuke"`, constructor
  `(embedder: Embedder, top_k: int = 10, denoise_lambda: float = 1.3)`. Task 5's configs use it.

- [ ] **Step 1: Write the failing tests**

`tests/adapters/test_hcuke_scorer.py`:

```python
import pytest

from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.hcuke import HCUKEScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class LookupEmbedder(Embedder):
    """Test double: fixed vector per exact text; `default` otherwise."""

    def __init__(self, mapping: dict[str, tuple[float, ...]], default: tuple[float, ...]):
        self.mapping = mapping
        self.default = default

    @property
    def dim(self) -> int:
        return len(self.default)

    def embed(self, texts):
        return [self.mapping.get(t, self.default) for t in texts]


def _two_sentence_fixture():
    u0 = make_unit(id="d:u0", document_id="d", text="alpha beta.", order=0)
    u1 = make_unit(id="d:u1", document_id="d", text="gamma alpha.", order=1)
    mentions = [
        make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        make_mention(surface="gamma", unit_id="d:u1", span=(0, 5)),
        make_mention(surface="alpha", unit_id="d:u1", span=(6, 11)),
    ]
    return [u0, u1], mentions


class TestHCUKEScorer(ScorerContract):
    def make_scorer(self) -> HCUKEScorer:
        return HCUKEScorer(embedder=HashingEmbedder(dim=16))

    def test_hand_computed_scores_on_two_sentence_document(self):
        # Vectors chosen so every cosine is exactly 0 or 1:
        #   H_d = H_s0 = H_alpha = H_beta = x = (1,0,0); H_s1 = H_gamma = y = (0,1,0).
        # Sentence weights (Eq. 3): softmax(1/1, 1/2) = (0.622459, 0.377541).
        # Global (Alg. 1): alpha = W(s0)*1*1 + W(s1)*0*0 = 0.622459
        #                  beta  = W(s0)*1*1 = 0.622459;  gamma = W(s1)*0*1 = 0.
        # First word positions: alpha 1, beta 2, gamma 3 ->
        #   W(c) = softmax(1, 1/2, 1/3) = (0.471709, 0.286106, 0.242184).
        # Local (Eq. 6): pair sims (a,b)=1, (a,g)=0, (b,g)=0 -> mu = 1/3;
        #   lambda=1.3 -> R_l(alpha) = R_l(beta) = (1 - 13/30) + (0 - 13/30)
        #   = 0.133333; R_l(gamma) = -0.866667.
        # Final (Eq. 7): alpha = 0.622459 * 0.133333 * 0.471709 = 0.039149
        #                beta  = 0.622459 * 0.133333 * 0.286106 = 0.023745
        #                gamma = 0 * -0.866667 * 0.242184       = -0.0
        units, mentions = _two_sentence_fixture()
        x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        embedder = LookupEmbedder(
            {"alpha beta.": x, "gamma alpha.": y, "alpha": x, "beta": x, "gamma": y},
            default=x,  # the joined document text also maps to x
        )
        scorer = HCUKEScorer(embedder=embedder, denoise_lambda=1.3)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, units)}
        assert salience["alpha"] == pytest.approx(0.039149, abs=1e-5)
        assert salience["beta"] == pytest.approx(0.023745, abs=1e-5)
        assert salience["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert salience["alpha"] > salience["beta"] > salience["gamma"]

    def test_earlier_first_occurrence_wins_when_semantics_identical(self):
        # denoise_lambda=0 isolates the position bias: identical vectors give
        # equal global and local scores, so only W(c) (Eq. 3) differs.
        unit = make_unit(id="d:u0", text="alpha beta", order=0)
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        embedder = LookupEmbedder({}, default=(1.0, 1.0, 0.0))
        scorer = HCUKEScorer(embedder=embedder, denoise_lambda=0.0)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert salience["alpha"] > salience["beta"]

    def test_global_significance_restricted_to_own_sentences(self):
        # gamma's only sentence is orthogonal to the document, so its global
        # significance — and with it the final score — is exactly zero, no
        # matter how central other sentences are.
        units, mentions = _two_sentence_fixture()
        x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        embedder = LookupEmbedder(
            {"alpha beta.": x, "gamma alpha.": y, "alpha": x, "beta": x, "gamma": y},
            default=x,
        )
        scorer = HCUKEScorer(embedder=embedder)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, units)}
        assert salience["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert salience["beta"] > salience["gamma"]

    def test_empty_units_yields_defined_scores_and_lexicographic_tie(self):
        # No units -> no sentence layer -> every global score is 0, so every
        # final score is (+/-)0.0: a genuine tie, broken lexicographically.
        scorer = HCUKEScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        mentions = [
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        ]
        scored = scorer.score(mentions, [])
        assert all(sm.salience == 0.0 for sm in scored)
        assert {sm.mention.surface for sm in scored if sm.selected} == {"alpha"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_hcuke_scorer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lattice.adapters.scorer.hcuke'`

- [ ] **Step 3: Implement the adapter**

`src/lattice/adapters/scorer/hcuke.py`:

```python
import math
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.core.vectors import cosine
from lattice.ports import Embedder, Scorer
from lattice.registry.registry import register


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


@register(Scorer, "hcuke")
class HCUKEScorer(Scorer):
    """HCUKE (Xu et al., Knowledge-Based Systems 304 (2024) 112511):
    hierarchical context-aware unsupervised keyphrase extraction, implemented
    per the paper's Algorithm 1 with the pipeline's units as the sentence
    level (pair with the "sentence" segmenter for paper-faithful runs).

    - Position weights (Eq. 3): W(x) = softmax of 1/position over candidates
      (1-based document word position of the first occurrence, SIFRank-style)
      and over sentences (1-based order).
    - Global significance (Alg. 1 lines 8-13): R_g(c) = sum over sentences s
      containing c of W(s) * cos(H_s, H_d) * cos(H_c, H_s). Note: Eq. (5)'s
      prose would apply W(s) twice (it already sits inside Eq. (4)); Algorithm
      1 and the §3.3 worked example apply it once — we follow Algorithm 1.
    - Local significance (Eq. 6): R_l(c_i) = sum over j != i of
      (cos(H_ci, H_cj) - lambda * mu), mu = mean pairwise candidate
      similarity. Self-pairs are excluded (the paper is ambiguous; a
      self-pair adds the same constant to every candidate).
    - Final score (Eq. 7): R(c) = R_g(c) * R_l(c) * W(c); top_k unique
      surfaces by (-score, surface).

    Documented deviations: candidates, sentences, and documents are embedded
    as whole strings through the injected Embedder (paper: BERT token vectors
    + max-pooling); candidates come from the injected Extractor (paper:
    CoreNLP POS regex); word positions use whitespace tokens (paper: CoreNLP
    tokens). denoise_lambda defaults to the paper's Inspec-tuned 1.3 (§4.2)."""

    def __init__(self, embedder: Embedder, top_k: int = 10, denoise_lambda: float = 1.3):
        self.embedder = embedder
        self.top_k = top_k
        self.denoise_lambda = denoise_lambda

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        surfaces = sorted({m.surface for m in mentions})
        document_text = "\n".join(unit.text for unit in units)
        vectors = self.embedder.embed(
            [document_text, *(unit.text for unit in units), *surfaces]
        )
        document_vector = vectors[0]
        unit_vectors = {u.id: v for u, v in zip(units, vectors[1 : 1 + len(units)])}
        surface_vectors = dict(zip(surfaces, vectors[1 + len(units) :]))

        first_position = self._first_word_positions(mentions, units)
        candidate_weight = dict(zip(surfaces, _softmax(
            [1.0 / first_position[s] if s in first_position else 0.0 for s in surfaces]
        )))
        sentence_weight = dict(zip(
            (u.id for u in units), _softmax([1.0 / (u.order + 1) for u in units])
        ))

        units_of_surface: dict[str, set[str]] = {}
        for m in mentions:
            if m.unit_id in unit_vectors:
                units_of_surface.setdefault(m.surface, set()).add(m.unit_id)
        global_sig = {
            surface: sum(
                sentence_weight[unit_id]
                * cosine(unit_vectors[unit_id], document_vector)
                * cosine(surface_vectors[surface], unit_vectors[unit_id])
                for unit_id in sorted(units_of_surface.get(surface, ()))
            )
            for surface in surfaces
        }

        pair_sim = {
            (a, b): cosine(surface_vectors[a], surface_vectors[b])
            for i, a in enumerate(surfaces)
            for b in surfaces[i + 1 :]
        }
        mu = sum(pair_sim.values()) / len(pair_sim) if pair_sim else 0.0
        local_sig = {
            s: sum(
                pair_sim[(min(s, other), max(s, other))] - self.denoise_lambda * mu
                for other in surfaces
                if other != s
            )
            for s in surfaces
        }

        salience = {
            s: global_sig[s] * local_sig[s] * candidate_weight[s] for s in surfaces
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]

    @staticmethod
    def _first_word_positions(
        mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> dict[str, int]:
        """1-based document word position of each surface's first occurrence.
        Mentions pointing at units not present in `units` are skipped; a
        surface with no resolvable position falls back to a zero position
        score in the softmax (uniform weight in the degenerate case)."""
        unit_by_id = {u.id: u for u in units}
        unit_offset: dict[str, int] = {}
        offset = 0
        for u in sorted(units, key=lambda u: u.order):
            unit_offset[u.id] = offset
            offset += len(u.text.split())
        positions: dict[str, int] = {}
        for m in mentions:
            unit = unit_by_id.get(m.unit_id)
            if unit is None:
                continue
            pos = unit_offset[unit.id] + len(unit.text[: m.span[0]].split()) + 1
            positions[m.surface] = min(pos, positions.get(m.surface, pos))
        return positions
```

In `src/lattice/adapters/__init__.py`, change the scorer import line to:

```python
from lattice.adapters.scorer import embedding_cosine, frequency, hcuke, mderank  # noqa: F401
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/adapters/test_hcuke_scorer.py -q`
Expected: 7 passed (3 contract + 4 new)

- [ ] **Step 5: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add src/lattice/adapters/scorer/hcuke.py src/lattice/adapters/__init__.py tests/adapters/test_hcuke_scorer.py
git commit -m "feat: add HCUKE scorer adapter (KBS 2024, Algorithm 1)"
```

---

### Task 5: M2b sweep config + end-to-end tests

**Files:**
- Create: `configs/m2b-sweep.toml`
- Create: `tests/harness/test_m2b_e2e.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4 by registered name (`"sentence"`, `"mderank"`,
  `"hcuke"`); the existing harness (`ExperimentConfig`, `run_experiment`), sweep runner,
  mini-Inspec fixture at `tests/fixtures/mini_inspec`.
- Produces: the M2b exit-criteria sweep config; e2e regression tests.

- [ ] **Step 1: Write the sweep config**

`configs/m2b-sweep.toml`:

```toml
# M2b sweep (spec §11): all scorers under identical conditions — sentence
# units, noun-chunk candidates, MiniLM embeddings — on the Inspec test split.
# The sentence segmenter is the base so every scorer sees the same candidate
# set (fair comparison) and HCUKE gets its sentence level.

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
]
```

- [ ] **Step 2: Write the e2e tests (failing only if wiring is broken — run them)**

`tests/harness/test_m2b_e2e.py`:

```python
import pytest

from lattice.harness.runner import ExperimentConfig, run_experiment

MINI_ROOT = "tests/fixtures/mini_inspec"

METRIC_KEYS = {
    "precision@5", "recall@5", "f1@5",
    "precision@10", "recall@10", "f1@10",
    "precision@15", "recall@15", "f1@15",
}


def _config(extractor: dict, embedder: dict, scorer: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "segmenter": {"name": "sentence"},
            "extractor": extractor,
            "scorer": scorer,
            "resolver": {"name": "exact-label"},
            "relation_inducer": {"name": "co-occurrence"},
            "graph_integrator": {"name": "in-memory"},
            "embedder": embedder,
            "dataset": {"name": "inspec", "params": {"root": MINI_ROOT}},
            "document_metrics": [{"name": "f1-at-k"}],
        }
    )


def _pure(scorer: str) -> ExperimentConfig:
    return _config(
        extractor={"name": "token", "params": {"min_length": 4}},
        embedder={"name": "hashing"},
        scorer={"name": scorer, "params": {"top_k": 15}},
    )


def _ml(scorer: str) -> ExperimentConfig:
    return _config(
        extractor={"name": "noun-chunk"},
        embedder={"name": "sentence-transformer"},
        scorer={"name": scorer, "params": {"top_k": 15}},
    )


@pytest.mark.parametrize("scorer", ["mderank", "hcuke"])
def test_m2b_scorer_pipeline_pure(scorer):
    """Both frontier scorers through the full harness with M1's pure adapters:
    proves registration and wiring without the ml stack."""
    report = run_experiment(_pure(scorer))
    assert report.errors == ()
    assert report.documents_processed == 3
    assert set(report.metrics["f1-at-k"]) == METRIC_KEYS


@pytest.mark.ml
@pytest.mark.parametrize("scorer", ["mderank", "hcuke"])
def test_m2b_scorer_real_ml_path(scorer):
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        report = run_experiment(_ml(scorer))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
    assert report.errors == ()
    # Quality thresholds live on the real Inspec benchmark (M2a lesson: a
    # 3-document fixture cannot hold one); this proves the real ML path runs
    # end-to-end with defined metrics.
    assert 0.0 <= report.metrics["f1-at-k"]["f1@15"] <= 1.0


@pytest.mark.ml
def test_hcuke_is_reproducible():
    pytest.importorskip("spacy")
    pytest.importorskip("sentence_transformers")
    try:
        assert run_experiment(_ml("hcuke")) == run_experiment(_ml("hcuke"))
    except OSError:
        pytest.skip("models not cached (run scripts/fetch_models.py)")
```

- [ ] **Step 3: Run the new tests (ml group is installed on this machine)**

Run: `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest tests/harness/test_m2b_e2e.py -q`
Expected: 5 passed (2 pure + 3 ml; ml ones skip only if models are missing — on this machine they are cached and must pass)

- [ ] **Step 4: Lint, full suite, commit**

Run: `uv run --no-sync ruff check && chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q`
Expected: ruff clean; suite green

```bash
git add configs/m2b-sweep.toml tests/harness/test_m2b_e2e.py
git commit -m "feat: add M2b sweep config and end-to-end tests"
```

---

## Exit criteria (orchestrator runs after Task 5 review clears)

1. Full suite green + ruff clean:
   `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync pytest -q && uv run --no-sync ruff check`
2. The real sweep (dataset already fetched in M2a; ~5–10 min on CPU — MDERank embeds one
   masked document per candidate):
   `chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null; uv run --no-sync python -m lattice.harness --sweep configs/m2b-sweep.toml reports/m2b`
   *(Corrected 2026-07-11: the as-built CLI is `python -m lattice.harness --sweep` per
   `src/lattice/harness/__main__.py`; `python -m lattice.harness.sweep` is a silent no-op —
   PLAN-DEFECT caught by the orchestrator at the exit gate.)*
3. Verify in `reports/`: four rows, zero errors in every row.
   - **mderank:** f1@10 in the amended spec band **[0.28, 0.38]** (paper: 0.3381 with BERT).
     Note it may land *below* embedding-cosine — that is what the paper reports on Inspec
     (spec §11 as amended 2026-07-10); it is a finding, not a bug.
   - **hcuke:** recorded with zero errors; sanity floor f1@10 ≥ the frequency row. Published
     0.4341 used BERT token max-pooling; our whole-string MiniLM deviation may land lower.
     If below the floor, STOP and report for adjudication before closing (do not tune).
     *(Resolved 2026-07-11, SUPERSEDED same day: real sweep initially gave hcuke f1@10 = 0.1309,
     below the floor. Plan-author's first-pass adjudication after diagnosing 3 real documents
     concluded this was purely a representation-fidelity cost (whole-phrase MiniLM vs. the
     paper's BERT token-max-pooling) and accepted it without a code change. The final
     whole-branch (opus) review caught what that first pass missed: `local_sig` excluded the
     self-pair term from Eq. 6's sum, but Algorithm 1 line 15's inner loop
     (`while ci in C do ... R_l ← R_l + (cosine(Hc,Hci) − λμ)`) has no guard excluding `ci = c`,
     so the self-comparison (cos=1) is included by design — and because the final score is a
     *product* `R_g·R_l·W` (Eq. 7), a constant added to R_l is NOT harmless: it contributes a
     term proportional to that candidate's own R_g·W, not a uniform shift, so it changes the
     ranking. The "Algorithm 1 arithmetic hand-verified twice" claim in the superseded text
     above was true of the excluded-self-term formula, not of Algorithm 1 itself — an honest
     mistake in the first adjudication, not a review failure once caught. Fixed in commit
     `99934bb` (self-term restored, docstring corrected, hand-computed test recomputed and
     independently re-derived by two separate reviewers); re-swept: **hcuke f1@10 = 0.1734**
     (up from 0.1309, a ~32% relative gain, confirming the fix's real effect). Still below the
     frequency floor (0.2398) and the published 0.4341 (BERT token-max-pooled) — *that* residual
     gap is the genuine representation-fidelity finding: the paper's local-significance formula
     depends on a similarity-space geometry (contextual token-level embeddings) that a
     whole-phrase general-purpose sentence embedder does not reproduce, and it now rests on a
     paper-faithful implementation rather than one with an undiagnosed formula deviation.
     The spec's actual §11 criterion for HCUKE — "documents its equations against the paper" —
     does not impose a numeric floor; this plan's stricter self-imposed floor did its job by
     routing a genuine, surprising result to adjudication rather than silent pass or ad hoc
     tuning. Accepted as a documented finding; M2b closes without further code changes to HCUKE.)*
   - **frequency / embedding-cosine:** near their M2a values (0.2387 / 0.3545); small shifts
     are expected from block→sentence segmentation changing spaCy chunking.
4. Record the table and verdicts in `.superpowers/sdd/progress-m2b.md`. Do NOT merge; stop
   after the final review fix wave and re-review.
