from collections.abc import Sequence

from lattice.core.types import Mention, ScoredMention, Unit
from lattice.ports import Scorer
from lattice.registry.registry import register


@register(Scorer, "passthrough")
class PassthroughScorer(Scorer):
    """Evaluation-protocol scorer (M3 spec §4.3): selects every mention at
    salience 1.0. Paired with the gold-mentions extractor so resolution
    metrics see every gold mention — never use it for salience experiments."""

    def score(
        self, mentions: Sequence[Mention], units: Sequence[Unit]
    ) -> list[ScoredMention]:
        return [
            ScoredMention(mention=m, salience=1.0, selected=True) for m in mentions
        ]
