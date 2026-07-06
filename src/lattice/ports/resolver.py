from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Document, Resolution, ScoredMention


class Resolver(ABC):
    """Maps selected mentions to canonical concepts, preserving identity
    across documents via its backing ConceptStore (spec §6). Stateful.
    Receives only mentions with selected=True; the orchestrator filters."""

    @abstractmethod
    def resolve(
        self, scored_mentions: Sequence[ScoredMention], document: Document
    ) -> list[Resolution]: ...
