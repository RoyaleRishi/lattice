from lattice.adapters.metric.hierarchy_sanity import HierarchySanity
from lattice.core.types import GraphSnapshot, Relation
from tests.contracts.metric_contract import MetricContract
from tests.helpers import make_concept


class TestHierarchySanityContract(MetricContract):
    def make_metric(self):
        return HierarchySanity()

    def make_ground_truth(self):
        return {}


def _edge(source: str, target: str, type: str = "IS_A") -> Relation:
    return Relation(
        type=type, source_id=source, target_id=target, confidence=1.0,
        provenance="d1",
    )


def _snapshot(*relations: Relation) -> GraphSnapshot:
    node_ids = sorted({r.source_id for r in relations} | {r.target_id for r in relations})
    return GraphSnapshot(
        concepts=tuple(make_concept(id=n, label=n) for n in node_ids),
        relations=tuple(relations),
    )


def _evaluate(*relations: Relation) -> dict[str, float]:
    return HierarchySanity().evaluate(_snapshot(*relations), {})


def test_empty_snapshot_is_all_zeros():
    result = HierarchySanity().evaluate(GraphSnapshot(concepts=(), relations=()), {})
    assert result == {
        "cycle-components": 0.0,
        "cycle-nodes": 0.0,
        "self-loops": 0.0,
        "max-depth": 0.0,
        "transitive-shortcuts": 0.0,
        "is-a-edges": 0.0,
    }


def test_two_cycle_detected():
    result = _evaluate(_edge("a", "b"), _edge("b", "a"))
    assert result["cycle-components"] == 1.0
    assert result["cycle-nodes"] == 2.0
    assert result["self-loops"] == 0.0


def test_self_loop_counted_separately_not_as_cycle_component():
    result = _evaluate(_edge("a", "a"), _edge("a", "b"))
    assert result["self-loops"] == 1.0
    assert result["cycle-components"] == 0.0
    assert result["max-depth"] == 1.0


def test_chain_depth_counts_edges():
    result = _evaluate(_edge("a", "b"), _edge("b", "c"))
    assert result["max-depth"] == 2.0
    assert result["transitive-shortcuts"] == 0.0


def test_triangle_shortcut_detected():
    result = _evaluate(_edge("a", "b"), _edge("b", "c"), _edge("a", "c"))
    assert result["transitive-shortcuts"] == 1.0


def test_diamond_has_exactly_one_shortcut():
    result = _evaluate(
        _edge("a", "b"), _edge("a", "c"), _edge("a", "d"),
        _edge("b", "d"), _edge("c", "d"),
    )
    assert result["transitive-shortcuts"] == 1.0  # only a->d


def test_edge_out_of_a_cycle_is_not_a_shortcut():
    # b->c's only alternative "path" re-uses the b->c edge itself via the
    # a<->b cycle; the mid-walk edge guard must reject it.
    result = _evaluate(_edge("a", "b"), _edge("b", "a"), _edge("b", "c"))
    assert result["transitive-shortcuts"] == 0.0


def test_depth_excludes_cycle_nodes():
    # a<->b is a cycle; the acyclic remainder is c->d (depth 1)
    result = _evaluate(
        _edge("a", "b"), _edge("b", "a"), _edge("b", "c"), _edge("c", "d")
    )
    assert result["cycle-nodes"] == 2.0
    assert result["max-depth"] == 1.0


def test_non_is_a_relations_are_ignored():
    result = _evaluate(
        _edge("a", "b", type="CO_OCCURS"), _edge("b", "a", type="CO_OCCURS")
    )
    assert result["is-a-edges"] == 0.0
    assert result["cycle-components"] == 0.0


def test_deep_chain_does_not_blow_the_stack():
    edges = [_edge(str(i), str(i + 1)) for i in range(2000)]
    result = _evaluate(*edges)
    assert result["max-depth"] == 2000.0
    assert result["cycle-components"] == 0.0
