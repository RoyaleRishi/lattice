from lattice.adapters.metric.edge_f1 import EdgeF1
from lattice.core.types import GraphSnapshot, Relation
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


class TestEdgeF1Contract(MetricContract):
    def make_metric(self):
        return EdgeF1()

    def make_ground_truth(self):
        return {"is_a_edges": [["a", "b"]], "terms": ["a", "b"]}


def _snapshot(edges: list[tuple[str, str]], extra_types: bool = False):
    labels = sorted({label for edge in edges for label in edge})
    concepts = tuple(make_concept(id=f"c:{label}", label=label) for label in labels)
    relations = tuple(
        Relation(type="IS_A", source_id=f"c:{a}", target_id=f"c:{b}",
                 confidence=1.0, provenance="d")
        for a, b in edges
    )
    if extra_types:
        relations += (
            Relation(type="CO_OCCURS", source_id=concepts[0].id,
                     target_id=concepts[-1].id, confidence=1.0, provenance="d"),
        )
    return GraphSnapshot(concepts=concepts, relations=relations)


GOLD = {"is_a_edges": [["olive oil", "oil"], ["canola", "oil"]], "terms": []}


def test_perfect_prediction():
    snapshot = _snapshot([("olive oil", "oil"), ("canola", "oil")])
    result = EdgeF1().evaluate(snapshot, GOLD)
    assert result == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "predicted_edges": 2.0, "gold_edges": 2.0,
    }


def test_partial_overlap():
    snapshot = _snapshot([("olive oil", "oil"), ("oil", "olive oil")])
    result = EdgeF1().evaluate(snapshot, GOLD)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_direction_matters():
    snapshot = _snapshot([("oil", "olive oil")])
    assert EdgeF1().evaluate(snapshot, GOLD)["f1"] == 0.0


def test_non_is_a_relations_are_ignored():
    snapshot = _snapshot([("olive oil", "oil")], extra_types=True)
    assert EdgeF1().evaluate(snapshot, GOLD)["predicted_edges"] == 1.0


def test_empty_everything_is_all_zeros_without_crashing():
    result = EdgeF1().evaluate(
        GraphSnapshot(concepts=(), relations=()), {"is_a_edges": [], "terms": []}
    )
    assert result == {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "predicted_edges": 0.0, "gold_edges": 0.0,
    }


def test_all_values_are_floats():
    snapshot = _snapshot([("olive oil", "oil")])
    assert all(isinstance(v, float) for v in EdgeF1().evaluate(snapshot, GOLD).values())
