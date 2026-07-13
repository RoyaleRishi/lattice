import re

from lattice.core.types import GraphSnapshot
from lattice.core.vectors import cosine
from lattice.ports import Metric
from lattice.registry.registry import register

_ARTICLE = re.compile(r"^(?:a|an|the)\s+")


def _normalize(label: str) -> str:
    """casefold -> strip one leading article -> strip one trailing 's'
    when the result stays >= 3 chars and the label doesn't end in 'ss'
    (M5 spec §4.1: "beatles"->"beatle", "glass"->"glass")."""
    norm = _ARTICLE.sub("", label.casefold().strip())
    if len(norm) > 3 and norm.endswith("s") and not norm.endswith("ss"):
        norm = norm[:-1]
    return norm


@register(Metric, "redundancy")
class Redundancy(Metric):
    """Intrinsic near-duplicate detection over the accreted graph (M5 spec
    §4.1): what the resolver failed to merge. Two concepts are
    near-duplicates when their stored embeddings' cosine >= threshold or
    their normalized labels collide. O(n²) pairwise scan — fine at this
    scale (top-k selection bounds concepts to the low thousands)."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def evaluate(
        self, snapshot: GraphSnapshot, ground_truth: dict[str, object]
    ) -> dict[str, float]:
        concepts = snapshot.concepts
        count = len(concepts)
        norms = [_normalize(concept.label) for concept in concepts]
        pairs = 0
        has_duplicate = [False] * count
        for i in range(count):
            for j in range(i + 1, count):
                near = (
                    norms[i] == norms[j]
                    or cosine(concepts[i].embedding, concepts[j].embedding)
                    >= self.threshold
                )
                if near:
                    pairs += 1
                    has_duplicate[i] = has_duplicate[j] = True
        return {
            "duplicate-rate": (sum(has_duplicate) / count) if count else 0.0,
            "near-duplicate-pairs": float(pairs),
            "concept-count": float(count),
        }
