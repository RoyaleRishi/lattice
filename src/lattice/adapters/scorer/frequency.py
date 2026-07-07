from collections import Counter
from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.ports import Scorer
from lattice.registry.registry import register


@register(Scorer, "frequency")
class FrequencyScorer(Scorer):
    """Trivial walking-skeleton scorer: salience = surface frequency
    normalized by the max frequency. Selects every mention of the top_k most
    frequent surfaces; ties break alphabetically for determinism."""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        if not mentions:
            return []
        counts = Counter(m.surface for m in mentions)
        max_count = max(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top_surfaces = {surface for surface, _ in ranked[: self.top_k]}
        return [
            ScoredMention(
                mention=m,
                salience=counts[m.surface] / max_count,
                selected=m.surface in top_surfaces,
            )
            for m in mentions
        ]
