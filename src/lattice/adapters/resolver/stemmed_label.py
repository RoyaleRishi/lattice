import uuid
from collections.abc import Sequence
from dataclasses import replace

import snowballstemmer

from lattice.core.types import Concept, Document, Resolution, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@register(Resolver, "stemmed-label")
class StemmedLabelResolver(Resolver):
    """Honest middle baseline (review finding): `ExactLabelResolver` with one
    change — the merge key is the Snowball-stemmed label rather than the raw
    lowercased label. Measurement on ConEL-2 showed `embedding-nn` at its
    shipped 0.90 threshold makes only 7 merges beyond exact-label out of 452
    resolutions, all morphological/spelling variants, and the Snowball
    stemmer reproduces 6 of those 7 (it does not fold British/American
    spelling, e.g. 'favourite colour' vs 'favorite color'). This resolver
    exists so that gain can be measured directly instead of attributed to
    embedding semantics.

    Stems with the same normalization as `f1_at_k.py` (`" ".join over
    stemWords(lower().split())`) so the two agree. The stored `Concept.label`
    is the *first surface form seen*, not the stem -- the stem is a matching
    key, not a display label -- while `Concept.id` is derived from the stem
    so identity is stable across surfaces that collapse to it.

    `ConceptStore.find_by_label` is keyed on `Concept.label` (the surface),
    so it cannot be used for stem lookup. `Concept.id` is instead a pure
    function of the stem, so this resolver recomputes it from the stem on
    every call and always confirms existence via `concept_store.get(...)` --
    the resolver-local `dict[str, str]` mapping stem -> concept id
    (`_concept_id_by_stem`) is consulted first only to skip recomputing the
    uuid5 hash; it is never the sole existence gate. That makes the cache
    correctness-inert: a miss (e.g. a fresh resolver after `Engine.load()`,
    which repopulates the concept store but not this cache) or a stale hit
    surviving a `ConceptStore.rollback()` (the orchestrator's
    `on_error="skip"` path) always falls through to a real store lookup, and
    a genuine miss there self-heals by recreating the same deterministic
    concept id rather than diverging. The residual risk is therefore not
    incorrect merges or duplicate ids, but that a recreated concept's
    `first_seen`/`updated_at` reset to the current document -- e.g. a
    concept whose only prior occurrence was rolled back away will look
    "newly first seen" on its next occurrence, same as it would for
    `exact-label`/`embedding-nn` losing their own store state."""

    def __init__(self, embedder: Embedder, concept_store: ConceptStore):
        self.embedder = embedder
        self.concept_store = concept_store
        self._stemmer = snowballstemmer.stemmer("english")
        self._concept_id_by_stem: dict[str, str] = {}

    def _stem(self, label: str) -> str:
        return " ".join(self._stemmer.stemWords(label.split()))

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
            stem = self._stem(label)
            concept_id = self._concept_id_by_stem.get(stem)
            if concept_id is None:
                concept_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"lattice:concept:{stem}"))
            # Cache miss must fall through to a real store lookup -- the
            # cache is a lookup-speed optimization, never the existence gate
            # (see class docstring: this is what keeps Engine.load() safe).
            existing = self.concept_store.get(concept_id)
            if existing is not None:
                self._concept_id_by_stem[stem] = concept_id
                updated = replace(existing, updated_at=document.id)
                self.concept_store.upsert(updated)
                resolutions.append(
                    Resolution(concept=updated, mention=scored_mention, is_new=False)
                )
            else:
                concept = Concept(
                    id=concept_id,
                    label=label,
                    embedding=vectors[label],
                    first_seen=document.id,
                    updated_at=document.id,
                )
                self.concept_store.upsert(concept)
                self._concept_id_by_stem[stem] = concept_id
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
