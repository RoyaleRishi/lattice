from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit


class Scorer(ABC):
    """Assigns salience to mentions and selects the keepers (spec §6).
    Units provide document context (some scorers need the full document)."""

    @abstractmethod
    def score(self, mentions: Sequence[Mention], units: Sequence[Unit]) -> list[ScoredMention]: ...
