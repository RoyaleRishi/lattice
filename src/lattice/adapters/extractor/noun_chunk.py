from collections.abc import Sequence

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register

_TRIM_POS = {"DET", "PRON"}


@register(Extractor, "noun-chunk")
class NounChunkExtractor(Extractor):
    """PoS-bounded noun-phrase candidates via spaCy noun chunks (M2 spec §6.1).
    Leading determiners/pronouns are trimmed; over-long and pronoun-only
    chunks are dropped. spaCy is imported lazily so this module is importable
    without the ml dependency group."""

    def __init__(self, model: str = "en_core_web_sm", max_tokens: int = 5):
        import spacy

        self.nlp = spacy.load(model, disable=["ner"])
        self.max_tokens = max_tokens

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            doc = self.nlp(unit.text)
            for chunk in doc.noun_chunks:
                tokens = list(chunk)
                while tokens and tokens[0].pos_ in _TRIM_POS:
                    tokens = tokens[1:]
                if not tokens or len(tokens) > self.max_tokens:
                    continue
                start = tokens[0].idx
                end = tokens[-1].idx + len(tokens[-1].text)
                mentions.append(
                    Mention(
                        surface=unit.text[start:end].lower(),
                        unit_id=unit.id,
                        span=(start, end),
                        context=unit.text,
                        head=chunk.root.text.lower(),
                        lemma=" ".join(t.lemma_ for t in tokens).lower(),
                    )
                )
        return mentions
