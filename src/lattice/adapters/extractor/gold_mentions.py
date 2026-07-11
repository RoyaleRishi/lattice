import json
from collections.abc import Sequence
from pathlib import Path

from lattice.core.types import Mention, Unit
from lattice.ports import Extractor
from lattice.registry.registry import register


@register(Extractor, "gold-mentions")
class GoldMentionExtractor(Extractor):
    """Evaluation-protocol extractor (M3 spec §4.2): emits the gold mention
    spans stored in the converted corpus JSONL, so resolution metrics are not
    contaminated by extraction errors. Must be configured with the same
    root/split as the mention-clusters dataset, and paired with the block
    segmenter (converters emit single-newline text: one unit per document)."""

    def __init__(self, root: str, split: str = "test"):
        path = Path(root) / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the corpus fetch script "
                "(scripts/fetch_ecbplus.py or scripts/fetch_conel2.py) first"
            )
        self._by_document: dict[str, dict] = {}
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                self._by_document[record["id"]] = record

    def extract(self, units: Sequence[Unit]) -> list[Mention]:
        mentions: list[Mention] = []
        for unit in units:
            record = self._by_document.get(unit.document_id)
            if record is None:
                raise ValueError(
                    f"document {unit.document_id!r} not in gold mention sidecar — "
                    "configure gold-mentions with the same root/split as the dataset"
                )
            if unit.text != record["text"]:
                raise ValueError(
                    f"unit text differs from stored document text for "
                    f"{unit.document_id!r} — use the block segmenter; converters emit "
                    "single-newline text so each document is exactly one unit"
                )
            for m in record["mentions"]:
                mentions.append(
                    Mention(
                        surface=m["surface"],
                        unit_id=unit.id,
                        span=(m["start"], m["end"]),
                        context=record["text"][max(0, m["start"] - 40):m["end"] + 40],
                    )
                )
        return mentions
