"""Contract every Embedder adapter must satisfy."""

from lattice.ports import Embedder


class EmbedderContract:
    def make_embedder(self) -> Embedder:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_one_vector_per_text(self):
        vectors = self.make_embedder().embed(["vector store", "encoder"])
        assert len(vectors) == 2

    def test_vectors_have_declared_dim(self):
        embedder = self.make_embedder()
        [vector] = embedder.embed(["vector store"])
        assert len(vector) == embedder.dim

    def test_embedding_is_deterministic(self):
        embedder = self.make_embedder()
        assert embedder.embed(["vector store"]) == embedder.embed(["vector store"])

    def test_empty_input_yields_empty_list(self):
        assert self.make_embedder().embed([]) == []
