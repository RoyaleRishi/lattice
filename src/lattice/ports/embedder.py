from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Cross-cutting port: text → fixed-dimension vector (spec §6)."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...
