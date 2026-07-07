from lattice.core.types import Document, Unit
from lattice.ports import Segmenter
from lattice.registry.registry import register


@register(Segmenter, "block")
class BlockSegmenter(Segmenter):
    """Splits document text into blocks on blank lines."""

    def segment(self, document: Document) -> list[Unit]:
        blocks = [b.strip() for b in document.text.split("\n\n")]
        blocks = [b for b in blocks if b]
        return [
            Unit(
                id=f"{document.id}:u{i}",
                document_id=document.id,
                text=block,
                order=i,
                kind="block",
            )
            for i, block in enumerate(blocks)
        ]
