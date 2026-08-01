import pytest

from lattice.adapters.metric.redundancy import Redundancy, _normalize
from lattice.config.factory import instantiate
from lattice.config.schema import AdapterSpec
from lattice.core.types import Concept, GraphSnapshot
from lattice.ports import Metric
from tests.contracts.metric_contract import MetricContract


class TestRedundancyContract(MetricContract):
    def make_metric(self):
        return Redundancy(threshold=0.9)

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
    result = Redundancy(threshold=0.9).evaluate(
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
    result = Redundancy(threshold=0.9).evaluate(
        _snapshot(
            _concept("c1", "the beatles", (1.0, 0.0)),
            _concept("c2", "beatles", (0.0, 1.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 1.0
    assert result["duplicate-rate"] == 1.0


def test_ss_guard_prevents_false_plural_collision():
    result = Redundancy(threshold=0.9).evaluate(
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
    result = Redundancy(threshold=0.9).evaluate(
        _snapshot(
            _concept("c1", "x", (0.0, 0.0)),
            _concept("c2", "y", (0.0, 0.0)),
        ),
        {},
    )
    assert result["near-duplicate-pairs"] == 0.0


def test_empty_snapshot_is_all_zeros():
    assert Redundancy(threshold=0.9).evaluate(_snapshot(), {}) == {
        "duplicate-rate": 0.0,
        "near-duplicate-pairs": 0.0,
        "concept-count": 0.0,
        "cosine-duplicate-pairs": 0.0,
        "label-duplicate-pairs": 0.0,
    }


def test_cosine_only_pair_increments_cosine_key_not_label_key():
    result = Redundancy(threshold=0.9).evaluate(
        _snapshot(
            _concept("c1", "alpha", (1.0, 0.0)),
            _concept("c2", "zeta", (1.0, 0.0)),
        ),
        {},
    )
    assert result["cosine-duplicate-pairs"] == 1.0
    assert result["label-duplicate-pairs"] == 0.0
    assert result["near-duplicate-pairs"] == 1.0


def test_label_only_pair_increments_label_key_not_cosine_key():
    result = Redundancy(threshold=0.9).evaluate(
        _snapshot(
            _concept("c1", "the beatles", (1.0, 0.0)),
            _concept("c2", "beatles", (0.0, 1.0)),
        ),
        {},
    )
    assert result["label-duplicate-pairs"] == 1.0
    assert result["cosine-duplicate-pairs"] == 0.0
    assert result["near-duplicate-pairs"] == 1.0


def test_omitting_the_threshold_raises_instead_of_taking_a_circular_default():
    """The hazard: the old signature defaulted to 0.9, which is the shipped
    `embedding-nn` resolver's threshold, so a config that forgot `params`
    silently measured the tautology (that is how
    reports/intervals/m5-conel2/interval-report.json was produced)."""
    with pytest.raises(ValueError, match="vacuous by construction"):
        Redundancy()


def test_config_omitting_params_fails_loudly_at_build_time():
    """Same hazard through the path it actually travels: a `[[metrics]]`
    entry with no `params`, instantiated by the composition root."""
    with pytest.raises(ValueError, match="`threshold` is required"):
        instantiate(Metric, AdapterSpec(name="redundancy"))


def test_cosine_criterion_is_vacuous_at_the_resolvers_own_threshold():
    """Why the default was circular, on a snapshot shaped like one a resolver
    at 0.90 leaves behind: no surviving pair can have cosine >= 0.90, so at a
    metric threshold of 0.90 `cosine-duplicate-pairs` is 0 whatever the graph
    contains, and only the label criterion carries signal. Below the
    resolver's threshold the same snapshot has a cosine pair to report."""
    snapshot = _snapshot(
        # cosine 0.8 — below a 0.90 resolver, so this pair survives the merge
        _concept("c1", "alpha", (1.0, 0.0, 0.0)),
        _concept("c2", "zeta", (0.8, 0.6, 0.0)),
        # a label collision the embedding cannot see, i.e. real redundancy
        _concept("c3", "the beatles", (0.0, 0.0, 1.0)),
        _concept("c4", "beatles", (0.0, 0.0, -1.0)),
    )
    circular = Redundancy(threshold=0.90).evaluate(snapshot, {})
    assert circular["cosine-duplicate-pairs"] == 0.0
    assert circular["label-duplicate-pairs"] == 1.0

    decoupled = Redundancy(threshold=0.60).evaluate(snapshot, {})
    assert decoupled["cosine-duplicate-pairs"] == 1.0
    assert decoupled["label-duplicate-pairs"] == 1.0
    assert decoupled["duplicate-rate"] > circular["duplicate-rate"]


def test_pair_meeting_both_criteria_increments_both_keys_and_counts_once():
    result = Redundancy(threshold=0.9).evaluate(
        _snapshot(
            _concept("c1", "the beatles", (1.0, 0.0)),
            _concept("c2", "beatles", (1.0, 0.0)),
        ),
        {},
    )
    assert result["cosine-duplicate-pairs"] == 1.0
    assert result["label-duplicate-pairs"] == 1.0
    assert result["near-duplicate-pairs"] == 1.0
