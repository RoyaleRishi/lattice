from lattice.adapters.embedder.hashing import HashingEmbedder
from lattice.adapters.scorer.mderank import MDERankScorer
from lattice.ports import Embedder
from tests.contracts.scorer_contract import ScorerContract
from tests.helpers import make_mention, make_unit


class BatchRecordingEmbedder(Embedder):
    """Test double: hashing embedder that records each embed() batch."""

    def __init__(self):
        self.inner = HashingEmbedder(dim=16)
        self.batches: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self.inner.dim

    def embed(self, texts):
        batch = list(texts)
        self.batches.append(batch)
        return self.inner.embed(batch)


class TestMDERankScorer(ScorerContract):
    def make_scorer(self) -> MDERankScorer:
        return MDERankScorer(embedder=HashingEmbedder(dim=16))

    def test_one_batch_of_document_plus_masked_variants(self):
        embedder = BatchRecordingEmbedder()
        scorer = MDERankScorer(embedder=embedder)
        unit = make_unit(id="d:u0", text="vector store holds a vector")
        mentions = [
            make_mention(surface="vector", unit_id="d:u0", span=(0, 6)),
            make_mention(surface="store", unit_id="d:u0", span=(7, 12)),
            make_mention(surface="vector", unit_id="d:u0", span=(21, 27)),
        ]
        scorer.score(mentions, [unit])
        assert len(embedder.batches) == 1  # single embed call per score()
        batch = embedder.batches[0]
        # [document, masked-per-unique-surface in sorted order: store, vector]
        assert batch[0] == "vector store holds a vector"
        assert batch[1] == "vector [MASK] holds a vector"
        assert batch[2] == "[MASK] store holds a [MASK]"

    def test_masking_a_central_candidate_scores_highest(self):
        # Masking the surface that constitutes most of the document moves the
        # document embedding far more than masking a peripheral one.
        scorer = self.make_scorer()
        unit = make_unit(id="d:u0", text="graph theory graph theory graph theory zebra")
        mentions = [
            make_mention(surface="graph theory", unit_id="d:u0", span=(0, 12)),
            make_mention(surface="graph theory", unit_id="d:u0", span=(13, 25)),
            make_mention(surface="graph theory", unit_id="d:u0", span=(26, 38)),
            make_mention(surface="zebra", unit_id="d:u0", span=(39, 44)),
        ]
        scored = {sm.mention.surface: sm.salience for sm in scorer.score(mentions, [unit])}
        assert scored["graph theory"] > scored["zebra"]

    def test_empty_units_yields_genuine_tie_broken_lexicographically(self):
        # With no units every masked variant equals the empty document; the
        # embedder maps "" to the zero vector, cosine(0, 0) is 0.0, so every
        # salience is exactly 1.0 — a genuine tie.
        scorer = MDERankScorer(embedder=HashingEmbedder(dim=16), top_k=1)
        mentions = [
            make_mention(surface="beta", unit_id="d:u0", span=(6, 10)),
            make_mention(surface="alpha", unit_id="d:u0", span=(0, 5)),
        ]
        scored = scorer.score(mentions, [])
        saliences = {sm.mention.surface: sm.salience for sm in scored}
        assert saliences["alpha"] == saliences["beta"] == 1.0
        assert {sm.mention.surface for sm in scored if sm.selected} == {"alpha"}
