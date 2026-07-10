from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.embedding_cosine import EmbeddingCosineScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class CountingEmbedder(Embedder):
    """Test double: hashing embedder that counts embed() texts."""

    def __init__(self):
        self.inner = HashingEmbedder(dim=16)
        self.texts_embedded: list[str] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts):
        self.texts_embedded.extend(texts)
        return self.inner.embed(texts)


class ConstantEmbedder(Embedder):
    """Test double: returns the same fixed nonzero vector for every text,
    forcing an exact cosine-salience tie across all surfaces."""

    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts):
        return [(1.0, 1.0, 1.0, 1.0) for _ in texts]


class TestEmbeddingCosineScorer(ScorerContract):
    def make_scorer(self) -> EmbeddingCosineScorer:
        return EmbeddingCosineScorer(embedder=HashingEmbedder(dim=16))

    def test_each_unique_surface_embedded_once(self):
        embedder = CountingEmbedder()
        scorer = EmbeddingCosineScorer(embedder=embedder)
        mentions = [
            make_mention(surface="vector store", span=(0, 12)),
            make_mention(surface="vector store", span=(20, 32)),
            make_mention(surface="encoder", span=(40, 47)),
        ]
        scorer.score(mentions, [make_unit(text="vector store text vector store encoder")])
        assert len(embedder.texts_embedded) == 3  # 1 document + 2 unique surfaces

    def test_document_similar_surface_scores_highest(self):
        scorer = self.make_scorer()
        unit = make_unit(text="vector store")
        mentions = [
            make_mention(surface="vector store", unit_id=unit.id, span=(0, 12)),
            make_mention(surface="zzz unrelated", unit_id=unit.id, span=(0, 3)),
        ]
        scored = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert scored["vector store"] > scored["zzz unrelated"]

    def test_top_k_selects_unique_surfaces_with_lexicographic_ties(self):
        scorer = EmbeddingCosineScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        unit = make_unit(text="alpha beta")
        mentions = [
            make_mention(surface="alpha", unit_id=unit.id, span=(0, 5)),
            make_mention(surface="beta", unit_id=unit.id, span=(6, 10)),
        ]
        scored = scorer.score(mentions, [unit])
        assert sum(1 for sm in scored if sm.selected) == 1

    def test_top_k_breaks_genuine_salience_tie_lexicographically(self):
        scorer = EmbeddingCosineScorer(embedder=ConstantEmbedder(), top_k=1)
        unit = make_unit(text="alpha beta")
        mentions = [
            make_mention(surface="beta", unit_id=unit.id, span=(6, 10)),
            make_mention(surface="alpha", unit_id=unit.id, span=(0, 5)),
        ]
        scored = scorer.score(mentions, [unit])
        saliences = {sm.mention.surface: sm.salience for sm in scored}
        assert saliences["alpha"] == saliences["beta"]  # genuine tie, not a fluke
        selected = {sm.mention.surface for sm in scored if sm.selected}
        assert selected == {"alpha"}

    def test_empty_units_yields_defined_scores(self):
        scorer = self.make_scorer()
        mentions = [make_mention(surface="alpha", span=(0, 5))]
        scored = scorer.score(mentions, [])
        assert len(scored) == 1
        assert scored[0].salience == 0.0  # empty document embeds to the zero vector
