import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace

import snowballstemmer

from lattice.adapters.resolver._store_backed import StoreBackedResolver
from lattice.core.types import Concept, Document, Resolution, ResolverState, ScoredMention
from lattice.ports import ConceptStore, Embedder, Resolver
from lattice.registry.registry import register


@dataclass(frozen=True, slots=True)
class _StemToken:
    """Combined checkpoint token: the store's token plus the stem-cache epoch."""

    store_token: object
    epoch: int


@register(Resolver, "stemmed-label")
class StemmedLabelResolver(StoreBackedResolver):
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
    every call and always confirms existence via `_store.get(...)`. The
    resolver-local `dict[str, str]` (`_concept_id_by_stem`) is consulted
    first only to skip recomputing the uuid5 hash.

    The cache is covered by the resolver lifecycle (ADR-0001/0002): it is
    undo-logged alongside the store checkpoint (first-touch prior value per
    stem), reported in `snapshot().private`, rebuilt from `private` on
    `restore()` (a missing key means an empty cache, which is valid since
    lookups fall through to the store), and cleared on `reset()`."""

    def __init__(self, embedder: Embedder, concept_store: ConceptStore):
        super().__init__(embedder, concept_store)
        self._stemmer = snowballstemmer.stemmer("english")
        self._concept_id_by_stem: dict[str, str] = {}
        # undo log: stem -> prior id (None = was absent), None when no checkpoint is open
        self._stem_log: dict[str, str | None] | None = None
        self._epoch = 0  # bumped whenever a token stops being valid

    def _remember(self, stem: str, concept_id: str) -> None:
        # first touch since the checkpoint records the prior value
        if self._stem_log is not None and stem not in self._stem_log:
            self._stem_log[stem] = self._concept_id_by_stem.get(stem)
        self._concept_id_by_stem[stem] = concept_id

    def _invalidate(self) -> None:
        self._stem_log = None
        self._epoch += 1

    def checkpoint(self) -> object:
        store_token = super().checkpoint()
        self._epoch += 1  # supersedes any earlier token
        self._stem_log = {}
        return _StemToken(store_token, self._epoch)

    def rollback(self, token: object) -> None:
        if (
            not isinstance(token, _StemToken)
            or token.epoch != self._epoch
            or self._stem_log is None
        ):
            raise ValueError("stale, superseded or already-used resolver checkpoint")
        super().rollback(token.store_token)
        for stem, prior in reversed(self._stem_log.items()):
            if prior is None:
                self._concept_id_by_stem.pop(stem, None)
            else:
                self._concept_id_by_stem[stem] = prior
        self._invalidate()  # a token is single-use

    def snapshot(self) -> ResolverState:
        base = super().snapshot()
        return ResolverState(
            base.concepts, {"concept_id_by_stem": dict(self._concept_id_by_stem)}
        )

    def restore(self, state: ResolverState) -> None:
        super().restore(state)
        self._concept_id_by_stem = dict(state.private.get("concept_id_by_stem", {}))
        self._invalidate()

    def reset(self) -> None:
        super().reset()
        self._concept_id_by_stem = {}
        self._invalidate()

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
            existing = self._store.get(concept_id)
            if existing is not None:
                self._remember(stem, concept_id)
                updated = replace(existing, updated_at=document.id)
                self._store.upsert(updated)
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
                self._store.upsert(concept)
                self._remember(stem, concept_id)
                resolutions.append(
                    Resolution(concept=concept, mention=scored_mention, is_new=True)
                )
        return resolutions
