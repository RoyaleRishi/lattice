from lattice.adapters.metric.label_f1 import LabelF1
from lattice.core.types import GraphSnapshot
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


def snapshot_of(*labels: str) -> GraphSnapshot:
    return GraphSnapshot(
        concepts=tuple(make_concept(id=f"c:{l}", label=l) for l in labels),
        relations=(),
    )


class TestLabelF1(MetricContract):
    def make_metric(self) -> LabelF1:
        return LabelF1()

    def make_ground_truth(self) -> dict:
        return {"concept_labels": ["vector", "store"]}

    def test_perfect_match_scores_one(self):
        result = LabelF1().evaluate(snapshot_of("vector", "store"), self.make_ground_truth())
        assert result == {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    def test_disjoint_labels_score_zero(self):
        result = LabelF1().evaluate(snapshot_of("apple", "zebra"), self.make_ground_truth())
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_partial_overlap(self):
        result = LabelF1().evaluate(snapshot_of("vector", "zebra"), self.make_ground_truth())
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5

    def test_empty_snapshot_scores_zero(self):
        result = LabelF1().evaluate(snapshot_of(), self.make_ground_truth())
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def test_gold_labels_compared_case_insensitively(self):
        result = LabelF1().evaluate(
            snapshot_of("vector", "store"), {"concept_labels": ["Vector", "STORE"]}
        )
        assert result["f1"] == 1.0
