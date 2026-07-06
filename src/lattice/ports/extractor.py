from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Mention, Unit


class Extractor(ABC):
    """Finds candidate concept mentions in units (spec §6)."""

    @abstractmethod
    def extract(self, units: Sequence[Unit]) -> list[Mention]: ...
