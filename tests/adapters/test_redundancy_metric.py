from lattice.adapters.metric.redundancy import Redundancy, _normalize
from lattice.core.types import Concept, GraphSnapshot
from tests.contracts.metric_contract import MetricContract


class TestRedundancyContract(MetricContract):
    def make_metric(self):
        return Redundancy()

    def make_ground_truth(self):
        return {}


def _concept(cid: str, label: str, embedding: tuple[float, ...]) -> Concept:
    return Concept(
        id=cid, label=label, embedding=embedding, first_seen="d1", updated_at="d1"
    )


def _snapshot(*concepts: Concept) -> GraphSnapshot:
    return GraphSnapshot(concepts=tuple(concepts), relations=())


def test_normalize_rules():
    assert _normalize("The Beatles") == "beatle"
    assert _normalize("beatles") == "beatle"
    assert _normalize("an apple") == "apple"
    assert _normalize("glass") == "glass"  # 'ss' guard: no plural strip
    assert _normalize("gas") == "gas"  # too short to strip
    assert _normalize("glas") == "gla"


def test_embedding_near_duplicates_counted():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "alpha", (1.0, 0.0)),
            _concept("c2", "beta", (1.0, 0.0)),
            _concept("c3", "gamma", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 1.0
    assert result["duplicate-rate"] == 2.0 / 3.0
    assert result["concept-count"] == 3.0


def test_label_collision_counts_even_with_orthogonal_embeddings():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "the beatles", (1.0, 0.0)),
            _concept("c2", "beatles", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 1.0
    assert result["duplicate-rate"] == 1.0


def test_ss_guard_prevents_false_plural_collision():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "glass", (1.0, 0.0)),
            _concept("c2", "glas", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 0.0
    assert result["duplicate-rate"] == 0.0


def test_threshold_is_respected():
    # cosine of these is ~0.9487: above 0.9, below 0.99
    a = (3.0, 1.0)
    b = (1.0, 0.0)
    snapshot = _snapshot(_concept("c1", "x", a), _concept("c2", "y", b))
    assert Redundancy(threshold=0.9).evaluate(snapshot, {})["near-duplicate-pairs"] == 1.0
    assert Redundancy(threshold=0.99).evaluate(snapshot, {})["near-duplicate-pairs"] == 0.0


def test_zero_vectors_never_match_by_embedding():
    result = Redundancy().evaluate(
        _snapshot(
            _concept("c1", "x", (0.0, 0.0)),
            _concept("c2", "y", (0.0, 0.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 0.0


def test_empty_snapshot_is_all_zeros():
    assert Redundancy().evaluate(_snapshot(), {}) == {
        "duplicate-rate": 0.0,
        "near-duplicate-pairs": 0.0,
        "concept-count": 0.0,
    }
