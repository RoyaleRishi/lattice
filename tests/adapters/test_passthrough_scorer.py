from lattice.adapters.scorer.passthrough import PassthroughScorer
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class TestPassthroughScorer(ScorerContract):
    def make_scorer(self) -> PassthroughScorer:
        return PassthroughScorer()

    def test_everything_selected_at_salience_one(self):
        unit = make_unit(id="d:u0", text="alpha beta gamma")
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="gamma", unit_id="d:u0", span=(11, 16)),
        ]
        scored = self.make_scorer().score(mentions, [unit])
        assert all(sm.selected and sm.salience == 1.0 for sm in scored)
        assert len(scored) == 3
