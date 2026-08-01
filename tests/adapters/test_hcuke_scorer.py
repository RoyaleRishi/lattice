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
        #   min-max over (0.622459, 0.622459, 0): alpha 1, beta 1, gamma 0.
        # First word positions: alpha 1, beta 2, gamma 3 -> W(c), which enters
        #   Eq. (7) RAW (it is already a softmax; see the adapter docstring):
        #   W(c) = softmax(1, 1/2, 1/3)
        #        = (e^1, e^0.5, e^(1/3)) / (e^1 + e^0.5 + e^(1/3))
        #        = (2.718282, 1.648721, 1.395612) / 5.762616
        #        = (0.471710, 0.286106, 0.242184).
        # Local (Eq. 6): pair sims (a,b)=1, (a,g)=0, (b,g)=0 with a unit
        #   diagonal. Algorithm 1 line 15's inner loop sums over ALL candidates
        #   including the self-pair (cos(H_c,H_c)=1), so the row totals are
        #   alpha 1+1+0 = 2, beta 1+1+0 = 2, gamma 0+0+1 = 1, and (§3.4, all
        #   n^2 ordered pairs) mu = (2+2+1)/3^2 = 5/9. Each R_l is its row total
        #   less n*lambda*mu = 3*1.3*5/9 = 2.166667:
        #   R_l(alpha) = 2 - 2.166667 = -0.166667
        #   R_l(beta)  = 2 - 2.166667 = -0.166667
        #   R_l(gamma) = 1 - 2.166667 = -1.166667
        #   min-max over those: alpha 1, beta 1, gamma 0. (That offset is the
        #   same for every candidate, so normalizing cancels it entirely --
        #   see test_scores_are_invariant_to_denoise_lambda.)
        # Final (Eq. 7) = normalized R_g * normalized R_l * raw W(c):
        #   alpha = 1 * 1 * 0.471710 = 0.471710
        #   beta  = 1 * 1 * 0.286106 = 0.286106
        #   gamma = 0 * 0 * 0.242184 = 0.0
        units, mentions = _two_sentence_fixture()
        x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        embedder = LookupEmbedder(
            {"alpha beta.": x, "gamma alpha.": y, "alpha": x, "beta": x, "gamma": y},
            default=x,  # the joined document text also maps to x
        )
        scorer = HCUKEScorer(embedder=embedder, denoise_lambda=1.3)
        salience = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, units)}
        assert salience["alpha"] == pytest.approx(0.471710, abs=1e-6)
        assert salience["beta"] == pytest.approx(0.286106, abs=1e-6)
        assert salience["gamma"] == pytest.approx(0.0, abs=1e-12)
        assert salience["alpha"] > salience["beta"] > salience["gamma"]

    def test_every_salience_is_non_negative(self):
        # The invariant the added normalization buys, and the one that broke:
        # un-normalized R_l sums n terms each shifted by -lambda*mu, so with the
        # paper's lambda=1.3 it is negative for most candidates, and a negative
        # factor in Eq. (7)'s product ranks the least central candidates first.
        # Normalized R_g and R_l live in [0, 1] and raw W(c) is a softmax, hence
        # strictly positive, so no product of the three can be negative.
        units, mentions = _two_sentence_fixture()  # 3 distinct candidates
        scorer = HCUKEScorer(embedder=HashingEmbedder(dim=16), denoise_lambda=1.3)
        scored = scorer.score(mentions, units)
        assert {sm.mention.surface for sm in scored} == {"alpha", "beta", "gamma"}
        assert all(sm.salience >= 0.0 for sm in scored)
        assert any(sm.salience > 0.0 for sm in scored)  # not vacuously all-zero

    def test_scores_are_invariant_to_denoise_lambda(self):
        # Eq. (6) subtracts the same n*lambda*mu offset from every candidate, so
        # min-max normalization cancels it exactly and lambda cannot move a
        # score. Documented in the adapter's deviations ledger: the paper's
        # Fig. 3 has F1 nearly — not perfectly — flat in lambda.
        units, mentions = _two_sentence_fixture()
        embedder = HashingEmbedder(dim=16)
        scores = [
            {
                sm.mention.surface: sm.salience
                for sm in HCUKEScorer(embedder=embedder, denoise_lambda=lam).score(
                    mentions, units
                )
            }
            for lam in (0.0, 1.3, 7.5)
        ]
        assert scores[1] == pytest.approx(scores[0])
        assert scores[2] == pytest.approx(scores[0])

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
        # No units -> no sentence layer -> every global score is 0, and the two
        # candidates are symmetric in R_l. Both normalized factors are therefore
        # constant, which is _min_max's degenerate case: each maps to 1.0 (a
        # factor with no ranking information must not collapse the product). No
        # resolvable word position leaves W(c) = softmax(0, 0) = (0.5, 0.5),
        # which passes through raw. So both scores are 1 * 1 * 0.5 = 0.5 — a
        # genuine tie, broken lexicographically.
        scorer = HCUKEScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        mentions = [
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        ]
        scored = scorer.score(mentions, [])
        assert all(sm.salience == pytest.approx(0.5) for sm in scored)
        assert {sm.mention.surface for sm in scored if sm.selected} == {"alpha"}
