"""Contract every Metric adapter must satisfy: float values, no crash on an
empty snapshot."""

from lattice.core.types import GraphSnapshot
from lattice.ports import Metric


class MetricContract:
    def make_metric(self) -> Metric:
        raise NotImplementedError("subclass must provide the adapter under test")

    def make_ground_truth(self) -> dict:
        raise NotImplementedError("subclass must provide matching ground truth")

    def test_returns_dict_of_floats(self):
        result = self.make_metric().evaluate(
            GraphSnapshot(concepts=(), relations=()), self.make_ground_truth()
        )
        assert result and all(isinstance(v, float) for v in result.values())

    def test_handles_empty_snapshot_without_crashing(self):
        self.make_metric().evaluate(
            GraphSnapshot(concepts=(), relations=()), self.make_ground_truth()
        )
