import math

import pytest

from lattice.adapters.scorer.promptrank import PromptBackend, PromptRankScorer
from lattice.ports import Scorer
from lattice.registry.registry import lookup
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class DictBackend(PromptBackend):
    """Test double: returns a preset log-prob per candidate (default -1.0),
    so the adapter's position/ranking logic is exercised without a model."""

    def __init__(self, scores: dict[str, float] | None = None, default: float = -1.0):
        self._scores = scores or {}
        self._default = default

    def score_candidates(self, document, candidates):
        return {c: self._scores.get(c, self._default) for c in candidates}


class TestPromptRankScorer(ScorerContract):
    def make_scorer(self) -> Scorer:
        return PromptRankScorer(backend=DictBackend())

    def test_registered_under_promptrank(self):
        assert lookup(Scorer, "promptrank") is PromptRankScorer

    def test_default_backend_is_lazy_not_built_at_construction(self):
        # No model/transformers import until score() is first called.
        scorer = PromptRankScorer()
        assert scorer._backend is None

    def test_earlier_candidate_wins_when_probability_ties(self):
        scorer = PromptRankScorer(top_k=1, backend=DictBackend({"alpha": -2.0, "beta": -2.0}))
        unit = make_unit(id="d:u0", text="alpha beta gamma alpha")  # doc_len=4
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        scored = {sm.mention.surface: sm for sm in scorer.score(mentions, [unit])}
        # K = 1.2e8 / 4**3 = 1_875_000.0
        # alpha pos=0 -> pos_norm=1_875_000.0     -> salience=-3_750_000.0
        # beta  pos=1 -> pos_norm=1_875_000.25    -> salience=-3_750_000.5
        assert scored["alpha"].salience == pytest.approx(-3_750_000.0)
        assert scored["beta"].salience == pytest.approx(-3_750_000.5)
        assert scored["alpha"].selected and not scored["beta"].selected

    def test_probability_dominates_position(self):
        scorer = PromptRankScorer(top_k=1, backend=DictBackend({"alpha": -10.0, "beta": -2.0}))
        unit = make_unit(id="d:u0", text="alpha beta gamma alpha")  # doc_len=4
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        scored = {sm.mention.surface: sm for sm in scorer.score(mentions, [unit])}
        # alpha: 1_875_000.0  * -10 = -18_750_000.0
        # beta:  1_875_000.25 * -2  =  -3_750_000.5
        assert scored["alpha"].salience == pytest.approx(-18_750_000.0)
        assert scored["beta"].salience == pytest.approx(-3_750_000.5)
        assert scored["beta"].selected and not scored["alpha"].selected

    def test_empty_units_disable_the_position_term(self):
        scorer = PromptRankScorer(backend=DictBackend({"alpha": -2.0}))
        m = make_mention(surface="alpha", unit_id="d:u0", span=(0, 5))
        scored = scorer.score([m], [])  # no units -> doc_len 0
        assert scored[0].salience == pytest.approx(-2.0)


@pytest.mark.ml
def test_real_t5_backend_is_finite_and_deterministic():
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    scorer_a = PromptRankScorer(top_k=2)  # default T5PromptBackend, t5-base
    unit = make_unit(
        id="d:u0", text="Graph neural networks learn node embeddings for graph data."
    )
    mentions = [
        make_mention(surface="graph neural networks", unit_id="d:u0", span=(0, 21)),
        make_mention(surface="node embeddings", unit_id="d:u0", span=(28, 43)),
    ]
    a = scorer_a.score(mentions, [unit])
    b = PromptRankScorer(top_k=2).score(mentions, [unit])
    assert {sm.mention.span for sm in a} == {m.span for m in mentions}  # every mention scored
    assert all(math.isfinite(sm.salience) for sm in a)
    assert [round(sm.salience, 4) for sm in a] == [round(sm.salience, 4) for sm in b]  # deterministic
