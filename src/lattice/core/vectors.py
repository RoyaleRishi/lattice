"""Shared pure vector math. Stdlib only (parent spec §5)."""

import math
from collections.abc import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors of equal length.

    Raises ValueError if `a` and `b` have different lengths (GC4: honest
    failure over silently truncating to the shorter vector and returning a
    plausible-looking wrong number). Returns 0.0 if either vector is a zero
    vector (deliberate; not an error)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
