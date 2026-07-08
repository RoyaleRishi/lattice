"""Contract every DocumentMetric adapter must satisfy."""

import pytest

from lattice.core.types import GraphDelta
from lattice.ports import DocumentMetric


class DocumentMetricContract:
    def make_metric(self) -> DocumentMetric:
        raise NotImplementedError("subclass must provide the adapter under test")

    def make_ground_truth(self) -> dict:
        raise NotImplementedError("subclass must provide matching ground truth")

    def make_deltas(self) -> list[GraphDelta]:
        raise NotImplementedError("subclass must provide deltas consistent with ground truth")

    def test_returns_dict_of_floats(self):
        result = self.make_metric().evaluate_documents(self.make_deltas(), self.make_ground_truth())
        assert result and all(isinstance(v, float) for v in result.values())

    def test_unknown_document_raises(self):
        deltas = [
            GraphDelta(document_id="not-in-ground-truth", concepts_added=(),
                       concepts_updated=(), relations_added=())
        ]
        with pytest.raises(ValueError):
            self.make_metric().evaluate_documents(deltas, self.make_ground_truth())
