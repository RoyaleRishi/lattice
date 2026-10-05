from lattice.core.types import ResolverState
from lattice.ports import ConceptStore, Embedder, Resolver


class StoreBackedResolver(Resolver):
    """Reusable base for resolvers whose only state is their ConceptStore
    (ADR-0001). The store is private (`_store`); the constructor param stays
    `concept_store` so factory injection by param name keeps working.
    Resolvers with extra private state override the lifecycle methods and
    call super() for the store part."""

    def __init__(self, embedder: Embedder, concept_store: ConceptStore):
        self.embedder = embedder
        self._store = concept_store

    def checkpoint(self) -> object:
        return self._store.checkpoint()

    def rollback(self, token: object) -> None:
        self._store.rollback(token)

    def snapshot(self) -> ResolverState:
        return ResolverState(tuple(sorted(self._store.all(), key=lambda c: c.id)), {})

    def restore(self, state: ResolverState) -> None:
        # state rebuilds from concepts alone, so private is ignored here
        self._store.reset()
        for concept in state.concepts:
            self._store.upsert(concept)

    def reset(self) -> None:
        self._store.reset()
