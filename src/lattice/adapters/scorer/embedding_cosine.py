from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.core.vectors import cosine
from lattice.ports import Embedder, Scorer
from lattice.registry.registry import register


@register(Scorer, "embedding-cosine")
class EmbeddingCosineScorer(Scorer):
    """KeyBERT/SIFRank-style baseline (M2 spec §6.3): salience is the cosine
    similarity between the candidate surface embedding and the whole-document
    embedding. Consumes the injected Embedder; each unique surface is embedded
    once per call."""

    def __init__(self, embedder: Embedder, top_k: int = 10):
        self.embedder = embedder
        self.top_k = top_k

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        document_text = "\n".join(unit.text for unit in units)
        surfaces = sorted({m.surface for m in mentions})
        document_vector, *candidate_vectors = self.embedder.embed([document_text, *surfaces])
        salience = {
            surface: cosine(vector, document_vector)
            for surface, vector in zip(surfaces, candidate_vectors)
        }
        ranked = sorted(salience.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m, salience=salience[m.surface], selected=m.surface in top_surfaces
            )
            for m in mentions
        ]
