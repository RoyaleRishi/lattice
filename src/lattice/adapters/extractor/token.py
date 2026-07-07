import re
from collections.abc import Sequence

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register

_WORD = re.compile(r"[A-Za-z][A-Za-z-]+")


@register(Extractor, "token")
class TokenExtractor(Extractor):
    """Trivial walking-skeleton extractor: every word of at least min_length
    characters is a candidate mention. Real noun-phrase extraction arrives in
    Milestone 2; this exists so the skeleton runs with zero NLP deps."""

    def __init__(self, min_length: int = 4):
        self.min_length = min_length

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            for match in _WORD.finditer(unit.text):
                word = match.group()
                if len(word) < self.min_length:
                    continue
                mentions.append(
                    Mention(
                        surface=word.lower(),
                        unit_id=unit.id,
                        span=(match.start(), match.end()),
                        context=unit.text,
                        head=word.lower(),
                        lemma=word.lower(),
                    )
                )
        return mentions
