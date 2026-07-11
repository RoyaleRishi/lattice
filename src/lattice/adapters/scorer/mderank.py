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
