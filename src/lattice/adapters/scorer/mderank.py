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
    for the MiniLM family. Documents exceeding the embedder's context window
    are not truncation-handled: 61 of 500 Inspec test abstracts exceed
    MiniLM's 256 word-piece window (median 158, max 497 word-pieces). Past
    the cutoff, masking a candidate is a no-op on the truncated input, so its
    masked-document embedding is (near-)identical to the unmasked one and its
    salience collapses toward zero (in the longest document, 22 such
    candidates score exactly 2.22e-16, versus a mean of 0.0852 for candidates
    whose first occurrence falls before the cutoff). This is detected, not
    prevented: `self.degenerate_surfaces` counts surfaces whose salience is
    `<= 1e-12` after each `score()` call, so a harness can surface it, but
    those candidates still receive a (degenerate) ranking rather than being
    dropped or raising — a partially-truncated document still yields a
    usable ranking for its early candidates."""

    def __init__(self, embedder: Embedder, top_k: int = 10, mask_token: str = "[MASK]"):
        self.embedder = embedder
        self.top_k = top_k
        self.mask_token = mask_token
        self.degenerate_surfaces: int = 0

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        self.degenerate_surfaces = 0
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
        self.degenerate_surfaces = sum(1 for value in salience.values() if value <= 1e-12)
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]
