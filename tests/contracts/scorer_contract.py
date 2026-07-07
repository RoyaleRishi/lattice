"""Contract every Scorer adapter must satisfy: score every mention it is
given (no drops, no additions), mark selection via the boolean flag."""

import math

from lattice.ports import Scorer
from tests.helpers import make_mention, make_unit


class ScorerContract:
    def make_scorer(self) -> Scorer:
        raise NotImplementedError("subclass must provide the adapter under test")

    def _fixture(self):
        unit = make_unit(id="d:u0", text="vector store vector")
        mentions = [
            make_mention(surface="vector", unit_id="d:u0", span=(0, 6)),
            make_mention(surface="store", unit_id="d:u0", span=(7, 12)),
            make_mention(surface="vector", unit_id="d:u0", span=(13, 19)),
        ]
        return mentions, [unit]

    def test_every_mention_scored_exactly_once(self):
        mentions, units = self._fixture()
        scored = self.make_scorer().score(mentions, units)
        assert sorted(sm.mention.span for sm in scored) == sorted(m.span for m in mentions)

    def test_salience_is_finite(self):
        mentions, units = self._fixture()
        assert all(math.isfinite(sm.salience) for sm in self.make_scorer().score(mentions, units))

    def test_empty_input_yields_empty_output(self):
        assert self.make_scorer().score([], []) == []
