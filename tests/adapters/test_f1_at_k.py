import pytest

from lattice.adapters.document_metric.f1_at_k import F1AtK
from tests.contracts.document_metric_contract import DocumentMetricContract
from tests.helpers import make_delta


class TestF1AtK(DocumentMetricContract):
    def make_metric(self) -> F1AtK:
        return F1AtK(ks=[5])

    def make_ground_truth(self) -> dict:
        return {"keyphrases_by_document": {"d1": ["vector store", "encoder"]}}

    def make_deltas(self):
        return [make_delta("d1", [("vector store", 0.9), ("encoder", 0.8)])]

    def test_perfect_at_k(self):
        result = self.make_metric().evaluate_documents(self.make_deltas(), self.make_ground_truth())
        assert result == {"precision@5": 1.0, "recall@5": 1.0, "f1@5": 1.0}

    def test_stemming_matches_inflections(self):
        gt = {"keyphrases_by_document": {"d1": ["neural networks"]}}
        deltas = [make_delta("d1", [("neural network", 0.9)])]
        assert F1AtK(ks=[5]).evaluate_documents(deltas, gt)["f1@5"] == 1.0

    def test_k_truncates_by_salience_rank(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"]}}
        deltas = [make_delta("d1", [("beta", 0.9), ("alpha", 0.5)])]
        result = F1AtK(ks=[1]).evaluate_documents(deltas, gt)
        assert result["recall@1"] == 0.0  # only top-1 (beta) is kept

    def test_salience_ties_break_lexicographically(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"]}}
        deltas = [make_delta("d1", [("beta", 0.9), ("alpha", 0.9)])]
        assert F1AtK(ks=[1]).evaluate_documents(deltas, gt)["recall@1"] == 1.0

    def test_stem_collision_denominator_dedupes_before_truncating(self):
        # "neural network" and "neural networks" stem alike. Dedupe-then-
        # truncate (the literature protocol) collapses them into a single
        # ranked stem *before* taking the top-2, so "encoder" claims the
        # second slot and the denominator stays at 2 -- not 1, as it would
        # if truncation happened before the second stem-dedupe pass.
        gt = {"keyphrases_by_document": {"d1": ["neural networks"]}}
        deltas = [make_delta("d1", [
            ("neural network", 0.9),
            ("neural networks", 0.8),
            ("encoder", 0.7),
        ])]
        result = F1AtK(ks=[2]).evaluate_documents(deltas, gt)
        # ranked stems (salience desc): ["neural network" (0.9), "encod" (0.7)]
        # predicted = {"neural network", "encod"}; gold = {"neural network"}
        # tp=1 -> precision = 1/2 = 0.5, recall = 1/1 = 1.0, f1 = 2*.5*1/1.5 = 2/3
        assert result["precision@2"] == 0.5
        assert result["recall@2"] == 1.0
        assert result["f1@2"] == pytest.approx(2 / 3)

    def test_macro_average_over_documents(self):
        gt = {"keyphrases_by_document": {"d1": ["alpha"], "d2": ["beta"]}}
        deltas = [
            make_delta("d1", [("alpha", 1.0)]),   # f1 = 1.0
            make_delta("d2", [("gamma", 1.0)]),   # f1 = 0.0
        ]
        assert F1AtK(ks=[5]).evaluate_documents(deltas, gt)["f1@5"] == 0.5

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="no documents"):
            F1AtK().evaluate_documents([], self.make_ground_truth())

    def test_missing_ground_truth_key_raises(self):
        with pytest.raises(ValueError, match="keyphrases_by_document"):
            F1AtK().evaluate_documents(self.make_deltas(), {})
