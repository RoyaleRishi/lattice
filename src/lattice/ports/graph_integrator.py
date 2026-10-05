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

    @abstractmethod
    def checkpoint(self) -> object:
        """Mark the current state and return an opaque token (ADR-0002).
        Starts a fresh undo log and implicitly commits any previous one."""
        ...

    @abstractmethod
    def rollback(self, token: object) -> None:
        """Restore state exactly as of the checkpoint that issued `token`
        (ADR-0002). Raises ValueError if the token isn't the current open
        checkpoint (stale, superseded, or already rolled back)."""
        ...
