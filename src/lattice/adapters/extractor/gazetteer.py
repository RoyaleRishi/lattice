import re
from collections.abc import Sequence
from pathlib import Path

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register


@register(Extractor, "gazetteer")
class GazetteerExtractor(Extractor):
    """Dictionary extractor (M4 spec §4.5): case-insensitive, whole-word,
    longest-match scan of a fixed term list against unit text. Boundaries
    are non-word-char lookarounds rather than \\b because terms may contain
    hyphens and punctuation. Must be configured with the same root/gold as
    the taxonomy dataset."""

    def __init__(self, root: str, gold: str):
        path = Path(root) / gold / "terms.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run --no-sync python "
                "scripts/fetch_texeval.py` first"
            )
        terms = [line.strip() for line in path.read_text().splitlines()]
        terms = [t for t in terms if t]
        alternation = "|".join(
            re.escape(t) for t in sorted(terms, key=lambda t: (-len(t), t))
        )
        self._regex = re.compile(
            rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE
        )

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            for match in self._regex.finditer(unit.text):
                start, end = match.span()
                mentions.append(
                    Mention(
                        surface=match.group(0),
                        unit_id=unit.id,
                        span=(start, end),
                        context=unit.text[max(0, start - 40) : end + 40],
                    )
                )
        return mentions
