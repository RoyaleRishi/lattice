from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Document, Resolution, ResolverState, ScoredMention


class Resolver(ABC):
    """Maps selected mentions to canonical concepts, preserving identity
    across documents via its backing ConceptStore (spec §6). Stateful.
    Receives only mentions with selected=True; the orchestrator filters.

    The resolver OWNS its ConceptStore. The lifecycle methods below cover the
    store plus any resolver-private state, so callers never reach into the
    store directly (ADR-0001)."""

    @abstractmethod
    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]: ...

    @abstractmethod
    def checkpoint(self) -> object:
        """Open an undo log and return an opaque token. A new checkpoint
        implicitly commits the previous one. Rolling back a stale,
        superseded or already-used token raises ValueError (ADR-0002)."""
        ...

    @abstractmethod
    def rollback(self, token: object) -> None:
        """Undo everything since `token` (store and private state)."""
        ...

    @abstractmethod
    def snapshot(self) -> ResolverState:
        """Concepts (sorted by id) plus JSON-serializable private state."""
        ...

    @abstractmethod
    def restore(self, state: ResolverState) -> None:
        """Replace ALL state with `state`. With private={} this must still
        work for any resolver whose state can be rebuilt from the concepts
        (ADR-0003 migration relies on it); otherwise raise a clear
        ValueError."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Empty the store and any private state."""
        ...
