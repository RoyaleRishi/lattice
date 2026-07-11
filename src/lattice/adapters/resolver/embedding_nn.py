import uuid
from collections.abc import Sequence
from dataclasses import replace

from lattice.core.types import Concept, Document, Resolution, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@register(Resolver, "embedding-nn")
class EmbeddingNNResolver(Resolver):
    """M3 resolver (spec §4.1): exact-label short-circuit, then embedding
    nearest-neighbour merge at `threshold` cosine similarity, else create a
    new concept. Concept embeddings are fixed at creation (no centroid
    updates in M3 — documented deferral). One embed batch per document;
    mentions resolve in input order, so later mentions can merge into
    concepts created earlier in the same document (stream semantics)."""

    def __init__(
        self, embedder: Embedder, concept_store: ConceptStore, threshold: float = 0.8
    ):
        self.embedder = embedder
        self.concept_store = concept_store
        self.threshold = threshold

    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]:
        if not scored_mentions:
            return []
        labels = [sm.mention.surface.strip().lower() for sm in scored_mentions]
        unique = sorted(set(labels))
        vectors = dict(zip(unique, self.embedder.embed(unique)))
        resolutions: list[Resolution] = []
        for scored_mention, label in zip(scored_mentions, labels):
            existing = self.concept_store.find_by_label(label)
            if existing is None:
                hits = self.concept_store.nearest(vectors[label], k=1)
                if hits and hits[0][1] >= self.threshold:
                    existing = hits[0][0]
            if existing is not None:
                updated = replace(existing, updated_at=document.id)
                self.concept_store.upsert(updated)
                resolutions.append(
                    Resolution(concept=updated, mention=scored_mention, is_new=False)
                )
            else:
                concept = Concept(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{label}")),
                    label=label,
                    embedding=vectors[label],
                    first_seen=document.id,
                    updated_at=document.id,
                )
                self.concept_store.upsert(concept)
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
