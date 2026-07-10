import math

import pytest

pytestmark = pytest.mark.ml
pytest.importorskip("sentence_transformers")

from lattice.adapters.embedder.sentence_transformer import (  # noqa: E402
    SentenceTransformerEmbedder,
)
from tests.contracts.embedder_contract import EmbedderContract  # noqa: E402


class TestSentenceTransformerEmbedder(EmbedderContract):
    _instance = None

    def make_embedder(self) -> SentenceTransformerEmbedder:
        if TestSentenceTransformerEmbedder._instance is None:
            try:
                TestSentenceTransformerEmbedder._instance = SentenceTransformerEmbedder()
            except OSError:
                pytest.skip("all-MiniLM-L6-v2 not cached (run scripts/fetch_models.py)")
        return TestSentenceTransformerEmbedder._instance

    def test_dim_is_384_for_minilm(self):
        assert self.make_embedder().dim == 384

    def test_vectors_are_unit_normalized(self):
        [vector] = self.make_embedder().embed(["vector store"])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-5)

    def test_semantic_neighbors_beat_strangers(self):
        embedder = self.make_embedder()
        a, b, c = embedder.embed(["vector database", "vector store", "banana bread recipe"])
        from lattice.core.vectors import cosine

        assert cosine(a, b) > cosine(a, c)
