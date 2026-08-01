from abc import ABC, abstractmethod

from lattice.core.types import Concept


class ConceptStore(ABC):
    """Cross-cutting port: the memory backing the Resolver (spec §6).
    Stateful; reset() is the reproducibility contract (spec §4.2)."""

    @abstractmethod
    def upsert(self, concept: Concept) -> None: ...

    @abstractmethod
    def get(self, concept_id: str) -> Concept | None: ...

    @abstractmethod
    def find_by_label(self, label: str) -> Concept | None: ...

    @abstractmethod
    def nearest(
        self, embedding: tuple[float, ...], k: int = 1
    ) -> list[tuple[Concept, float]]: ...

    @abstractmethod
    def all(self) -> list[Concept]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def checkpoint(self) -> object:
        """Return an opaque snapshot token of the current state, for
        rollback() to restore later (the skip-path recovery in
        Orchestrator.process)."""
        ...

    @abstractmethod
    def rollback(self, token: object) -> None:
        """Restore state to a token previously returned by checkpoint()."""
        ...
