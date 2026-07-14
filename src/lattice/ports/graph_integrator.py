from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import GraphSnapshot, Relation, Resolution


class GraphIntegrator(ABC):
    """Applies concepts and relations into the accreting graph (spec §6).
    Stateful; snapshot()/reset() are the reproducibility contract (spec §4.2)."""

    @abstractmethod
    def apply(self, resolutions: Sequence[Resolution], relations: Sequence[Relation]) -> None: ...

    @abstractmethod
    def snapshot(self) -> GraphSnapshot: ...

    @abstractmethod
    def restore(self, snapshot: GraphSnapshot) -> None:
        """Replace internal state with the snapshot's contents (M6 spec §3:
        the persistence hook — Engine.load hands back a saved graph)."""
        ...

    @abstractmethod
    def reset(self) -> None: ...
