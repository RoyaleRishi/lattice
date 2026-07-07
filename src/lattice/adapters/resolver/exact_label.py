import uuid
from collections.abc import Sequence
from dataclasses import replace

from lattice.core.types import Concept, Document, Resolution, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@register(Resolver, "exact-label")
class ExactLabelResolver(Resolver):
    """Trivial walking-skeleton resolver: normalizes the surface to lowercase
    and merges only on exact label match against the store. Embedding-NN and
    clustering resolvers arrive in Milestone 3 behind the same port."""

    def __init__(self, embedder: Embedder, concept_store: ConceptStore):
        self.embedder = embedder
        self.concept_store = concept_store

    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]:
        resolutions: list[Resolution] = []
        for scored_mention in scored_mentions:
            label = scored_mention.mention.surface.strip().lower()
            existing = self.concept_store.find_by_label(label)
            if existing is not None:
                updated = replace(existing, updated_at=document.id)
                self.concept_store.upsert(updated)
                resolutions.append(
                    Resolution(concept=updated, mention=scored_mention, is_new=False)
                )
            else:
                [embedding] = self.embedder.embed([label])
                concept = Concept(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{label}")),
                    label=label,
                    embedding=embedding,
                    first_seen=document.id,
                    updated_at=document.id,
                )
                self.concept_store.upsert(concept)
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
