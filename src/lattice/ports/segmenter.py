from abc import ABC, abstractmethod

from lattice.core.types import Document, Unit


class Segmenter(ABC):
    """Splits a document into ordered units (spec §6)."""

    @abstractmethod
    def segment(self, document: Document) -> list[Unit]: ...
