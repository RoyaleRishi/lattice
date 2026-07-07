import math

from lattice.adapters.embedder.hashing import HashingEmbedder
from tests.contracts.embedder_contract import EmbedderContract


class TestHashingEmbedder(EmbedderContract):
    def make_embedder(self) -> HashingEmbedder:
        return HashingEmbedder(dim=32)

    def test_dim_is_configurable(self):
        assert HashingEmbedder(dim=16).dim == 16

    def test_vectors_are_unit_normalized(self):
        [vector] = HashingEmbedder(dim=32).embed(["vector store"])
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)

    def test_different_texts_differ(self):
        embedder = HashingEmbedder(dim=32)
        [a] = embedder.embed(["vector store"])
        [b] = embedder.embed(["completely unrelated phrase"])
        assert a != b
