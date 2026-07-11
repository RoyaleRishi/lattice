import pytest

from lattice.adapters.document_metric.clustering import ClusteringMetric
from lattice.core.types import Concept, GraphDelta, Mention, Resolution, ScoredMention
from tests.contracts.document_metric_contract import DocumentMetricContract


def _delta(document_id: str, rows: list[tuple[str, tuple[int, int], str]]) -> GraphDelta:
    """Delta whose resolutions assign mention (surface, span) -> concept id."""
    resolutions = []
    for surface, span, concept_id in rows:
        mention = Mention(
            surface=surface, unit_id=f"{document_id}:u0", span=span, context=surface
        )
        scored = ScoredMention(mention=mention, salience=1.0, selected=True)
        concept = Concept(
            id=concept_id, label=surface, embedding=(1.0, 0.0),
            first_seen=document_id, updated_at=document_id,
        )
        resolutions.append(Resolution(concept=concept, mention=scored, is_new=True))
    return GraphDelta(
        document_id=document_id, concepts_added=(), concepts_updated=(),
        relations_added=(), resolutions=tuple(resolutions),
    )


class TestClusteringMetricContract(DocumentMetricContract):
    def make_metric(self) -> ClusteringMetric:
        return ClusteringMetric()

    def make_ground_truth(self) -> dict:
        return {"clusters_by_mention": {"d1:0-5": "g1", "d1:6-9": "g1"}}

    def make_deltas(self) -> list[GraphDelta]:
        return [_delta("d1", [("alpha", (0, 5), "k1"), ("beta", (6, 9), "k1")])]


class TestClusteringMetricValues:
    # Textbook example: gold G1={a,b,c}, G2={d,e}; predicted P1={a,b}, P2={c,d,e}.
    # B3 precision per mention: a=1, b=1, c=1/3, d=2/3, e=2/3 -> mean 11/15.
    # B3 recall per mention:    a=2/3, b=2/3, c=1/3, d=1, e=1 -> mean 11/15.
    # ARI: index=2, sum_pred=4, sum_gold=4, total=C(5,2)=10, expected=1.6,
    #      max=4 -> (2-1.6)/(4-1.6) = 1/6.
    GOLD = {
        "clusters_by_mention": {
            "d1:0-1": "G1", "d1:2-3": "G1", "d1:4-5": "G1", "d2:0-1": "G2", "d2:2-3": "G2",
        }
    }

    def _deltas(self) -> list[GraphDelta]:
        return [
            _delta("d1", [("a", (0, 1), "P1"), ("b", (2, 3), "P1"), ("c", (4, 5), "P2")]),
            _delta("d2", [("d", (0, 1), "P2"), ("e", (2, 3), "P2")]),
        ]

    def test_hand_computed_b3_and_ari(self):
        result = ClusteringMetric().evaluate_documents(self._deltas(), self.GOLD)
        assert result["b3-precision"] == pytest.approx(11 / 15)
        assert result["b3-recall"] == pytest.approx(11 / 15)
        assert result["b3-f1"] == pytest.approx(11 / 15)
        assert result["ari"] == pytest.approx(1 / 6)

    def test_perfect_clustering_scores_one(self):
        deltas = [
            _delta("d1", [("a", (0, 1), "X"), ("b", (2, 3), "X"), ("c", (4, 5), "X")]),
            _delta("d2", [("d", (0, 1), "Y"), ("e", (2, 3), "Y")]),
        ]
        result = ClusteringMetric().evaluate_documents(deltas, self.GOLD)
        assert result == {
            "b3-precision": 1.0, "b3-recall": 1.0, "b3-f1": 1.0, "ari": 1.0,
        }

    def test_single_predicted_cluster_has_perfect_recall(self):
        # gold {a,b} + {c}; predicted one cluster {a,b,c}:
        # precision = (2/3 + 2/3 + 1/3)/3 = 5/9; recall = 1; ari = 0.
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G1", "d1:4-5": "G2"}}
        deltas = [
            _delta("d1", [("a", (0, 1), "P"), ("b", (2, 3), "P"), ("c", (4, 5), "P")])
        ]
        result = ClusteringMetric().evaluate_documents(deltas, gold)
        assert result["b3-precision"] == pytest.approx(5 / 9)
        assert result["b3-recall"] == pytest.approx(1.0)
        assert result["ari"] == pytest.approx(0.0)

    def test_all_singletons_on_singleton_gold_is_perfect(self):
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G2"}}
        deltas = [_delta("d1", [("a", (0, 1), "P1"), ("b", (2, 3), "P2")])]
        result = ClusteringMetric().evaluate_documents(deltas, gold)
        assert result == {
            "b3-precision": 1.0, "b3-recall": 1.0, "b3-f1": 1.0, "ari": 1.0,
        }

    def test_coverage_mismatch_raises_both_directions(self):
        deltas = [_delta("d1", [("a", (0, 1), "P1"), ("z", (8, 9), "P1")])]
        gold = {"clusters_by_mention": {"d1:0-1": "G1", "d1:2-3": "G1"}}
        with pytest.raises(ValueError, match="coverage mismatch"):
            ClusteringMetric().evaluate_documents(deltas, gold)

    def test_empty_deltas_raise(self):
        with pytest.raises(ValueError, match="no documents"):
            ClusteringMetric().evaluate_documents([], self.GOLD)

    def test_missing_ground_truth_key_raises(self):
        with pytest.raises(ValueError, match="clusters_by_mention"):
            ClusteringMetric().evaluate_documents(self._deltas(), {"wrong": {}})
