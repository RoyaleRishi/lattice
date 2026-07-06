from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Document, Relation, Resolution, Unit


class RelationInducer(ABC):
    """Induces typed relations between resolved concepts (spec §6)."""

    @abstractmethod
    def induce(
        self,
        resolutions: Sequence[Resolution],
        units: Sequence[Unit],
        document: Document,
    ) -> list[Relation]: ...
