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
