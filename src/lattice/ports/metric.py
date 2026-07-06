from abc import ABC, abstractmethod

from lattice.core.types import GraphSnapshot


class Metric(ABC):
    """Harness port: scores a graph snapshot against ground truth (spec §6, §9)."""

    @abstractmethod
    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]: ...
