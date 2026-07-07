from lattice.adapters.scorer.frequency import FrequencyScorer
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


def _mentions(*surfaces: str):
    return [
        make_mention(surface=s, unit_id="d:u0", span=(i * 10, i * 10 + len(s)))
        for i, s in enumerate(surfaces)
    ]


class TestFrequencyScorer(ScorerContract):
    def make_scorer(self) -> FrequencyScorer:
        return FrequencyScorer()

    def test_most_frequent_surface_has_max_salience(self):
        scored = FrequencyScorer().score(
            _mentions("vector", "vector", "store"), [make_unit()]
        )
        by_surface = {sm.mention.surface: sm.salience for sm in scored}
        assert by_surface["vector"] == 1.0
        assert by_surface["store"] == 0.5

    def test_top_k_limits_selected_surfaces(self):
        scored = FrequencyScorer(top_k=1).score(
            _mentions("vector", "vector", "store"), [make_unit()]
        )
        selected = {sm.mention.surface for sm in scored if sm.selected}
        assert selected == {"vector"}

    def test_ties_break_alphabetically(self):
        scored = FrequencyScorer(top_k=1).score(
            _mentions("zebra", "apple"), [make_unit()]
        )
        selected = {sm.mention.surface for sm in scored if sm.selected}
        assert selected == {"apple"}
