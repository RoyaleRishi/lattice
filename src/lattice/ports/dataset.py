from abc import ABC, abstractmethod
from collections.abc import Iterator

from lattice.core.types import Document


class Dataset(ABC):
    """Harness port: yields documents in stream order plus ground truth (spec §6, §9).
    The ground-truth shape is metric-specific; each Metric documents what it expects."""

    @abstractmethod
    def documents(self) -> Iterator[Document]: ...

    @abstractmethod
    def ground_truth(self) -> dict[str, object]: ...
