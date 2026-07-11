import pytest

from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.hcuke import HCUKEScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class LookupEmbedder(Embedder):
    """Test double: fixed vector per exact text; `default` otherwise."""

    def __init__(self, mapping: dict[str, tuple[float, ...]], default: tuple[float, ...]):
        self.mapping = mapping
        self.default = default

    @property
    def dim(self) -> int:
        return len(self.default)

    def embed(self, texts):
        return [self.mapping.get(t, self.default) for t in texts]


def _two_sentence_fixture():
    u0 = make_unit(id="d:u0", document_id="d", text="alpha beta.", order=0)
    u1 = make_unit(id="d:u1", document_id="d", text="gamma alpha.", order=1)
    mentions = [
        make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        make_mention(surface="gamma", unit_id="d:u1", span=(0, 5)),
        make_mention(surface="alpha", unit_id="d:u1", span=(6, 11)),
    ]
    return [u0, u1], mentions


class TestHCUKEScorer(ScorerContract):
    def make_scorer(self) -> HCUKEScorer:
        return HCUKEScorer(embedder=HashingEmbedder(dim=16))

    def test_hand_computed_scores_on_two_sentence_document(self):
        # Vectors chosen so every cosine is exactly 0 or 1:
        #   H_d = H_s0 = H_alpha = H_beta = x = (1,0,0); H_s1 = H_gamma = y = (0,1,0).
        # Sentence weights (Eq. 3): softmax(1/1, 1/2) = (0.622459, 0.377541).
        # Global (Alg. 1): alpha = W(s0)*1*1 + W(s1)*0*0 = 0.622459
        #                  beta  = W(s0)*1*1 = 0.622459;  gamma = W(s1)*0*1 = 0.
        # First word positions: alpha 1, beta 2, gamma 3 ->
        #   W(c) = softmax(1, 1/2, 1/3) = (0.471709, 0.286106, 0.242184).
        # Local (Eq. 6): pair sims (a,b)=1, (a,g)=0, (b,g)=0 -> mu = 1/3;
        #   lambda=1.3 -> R_l(alpha) = R_l(beta) = (1 - 13/30) + (0 - 13/30)
        #   = 0.133333; R_l(gamma) = -0.866667.
        # Final (Eq. 7): alpha = 0.622459 * 0.133333 * 0.471709 = 0.039149
        #                beta  = 0.622459 * 0.133333 * 0.286106 = 0.023745
        #                gamma = 0 * -0.866667 * 0.242184       = -0.0
        units, mentions = _two_sentence_fixture()
        x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        embedder = LookupEmbedder(
            {"alpha beta.": x, "gamma alpha.": y, "alpha": x, "beta": x, "gamma": y},
            default=x,  # the joined document text also maps to x
        )
        scorer = HCUKEScorer(embedder=embedder, denoise_lambda=1.3)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, units)}
        assert salience["alpha"] == pytest.approx(0.039149, abs=1e-5)
        assert salience["beta"] == pytest.approx(0.023745, abs=1e-5)
        assert salience["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert salience["alpha"] > salience["beta"] > salience["gamma"]

    def test_earlier_first_occurrence_wins_when_semantics_identical(self):
        # denoise_lambda=0 isolates the position bias: identical vectors give
        # equal global and local scores, so only W(c) (Eq. 3) differs.
        unit = make_unit(id="d:u0", text="alpha beta", order=0)
        mentions = [
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
        ]
        embedder = LookupEmbedder({}, default=(1.0, 1.0, 0.0))
        scorer = HCUKEScorer(embedder=embedder, denoise_lambda=0.0)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert salience["alpha"] > salience["beta"]

    def test_global_significance_restricted_to_own_sentences(self):
        # gamma's only sentence is orthogonal to the document, so its global
        # significance — and with it the final score — is exactly zero, no
        # matter how central other sentences are.
        units, mentions = _two_sentence_fixture()
        x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        embedder = LookupEmbedder(
            {"alpha beta.": x, "gamma alpha.": y, "alpha": x, "beta": x, "gamma": y},
            default=x,
        )
        scorer = HCUKEScorer(embedder=embedder)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, units)}
        assert salience["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert salience["beta"] > salience["gamma"]

    def test_empty_units_yields_defined_scores_and_lexicographic_tie(self):
        # No units -> no sentence layer -> every global score is 0, so every
        # final score is (+/-)0.0: a genuine tie, broken lexicographically.
        scorer = HCUKEScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        mentions = [
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        ]
        scored = scorer.score(mentions, [])
        assert all(sm.salience == 0.0 for sm in scored)
        assert {sm.mention.surface for sm in scored if sm.selected} == {"alpha"}
