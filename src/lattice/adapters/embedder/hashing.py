import hashlib
import math
from collections.abc import Sequence

from lattice.ports import Embedder
from lattice.registry.registry import register


@register(Embedder, "hashing")
class HashingEmbedder(Embedder):
    """Deterministic character-trigram hashing embedder. Not semantically
    meaningful — a stand-in so the skeleton runs without a model download.
    A sentence-transformer adapter replaces it for real experiments (M2)."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dim
        padded = f" {text.lower()} "
        for i in range(max(len(padded) - 2, 0)):
            trigram = padded[i : i + 3]
            digest = hashlib.md5(trigram.encode()).hexdigest()
            vec[int(digest, 16) % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return tuple(vec)
