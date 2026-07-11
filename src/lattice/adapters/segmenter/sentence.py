import re

from lattice.core.types import Document, Unit
from lattice.ports import Segmenter
from lattice.registry.registry import register

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@register(Segmenter, "sentence")
class SentenceSegmenter(Segmenter):
    """Splits document text into sentences on terminal punctuation followed
    by whitespace. Deterministic stdlib rule providing HCUKE's sentence level
    (M2 spec §6.7); adequate for benchmark abstracts. Documented deviation
    from the paper's CoreNLP sentence tokenizer: no abbreviation handling."""

    def segment(self, document: Document) -> list[Unit]:
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(document.text)]
        sentences = [s for s in sentences if s]
        return [
            Unit(
                id=f"{document.id}:u{i}",
                document_id=document.id,
                text=sentence,
                order=i,
                kind="sentence",
            )
            for i, sentence in enumerate(sentences)
        ]
