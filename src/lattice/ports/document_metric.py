from abc import ABC, abstractmethod
from collections.abc import Sequence

from lattice.core.types import GraphDelta


class DocumentMetric(ABC):
    """Harness port: scores per-document pipeline output (the run's GraphDeltas)
    against per-document ground truth (M2 spec §5). Complements the
    snapshot-level Metric port."""

    @abstractmethod
    def evaluate_documents(
        self, deltas: Sequence[GraphDelta], ground_truth: dict[str, object]
    ) -> dict[str, float]: ...
