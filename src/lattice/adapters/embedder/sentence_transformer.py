from collections.abc import Sequence

from lattice.ports import Embedder
from lattice.registry.registry import register


@register(Embedder, "sentence-transformer")
class SentenceTransformerEmbedder(Embedder):
    """Real semantic embedder (M2 spec §6.2). Default all-MiniLM-L6-v2 for
    literature parity (parent spec §14). Deterministic inference; outputs are
    L2-normalized. sentence-transformers is imported lazily so this module is
    importable without the ml dependency group."""

    def __init__(
        self, model: str = "all-MiniLM-L6-v2", batch_size: int = 32, device: str = "cpu"
    ):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device=device)
        self._batch_size = batch_size
        get_dim = (
            getattr(self._model, "get_embedding_dimension", None)
            or self._model.get_sentence_embedding_dimension
        )
        self._dim = int(get_dim())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [tuple(float(x) for x in vector) for vector in vectors]
