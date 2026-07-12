import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "taxonomy")
class TaxonomyDataset(Dataset):
    """TExEval-2 taxonomy benchmark reader (M4 spec §4.4), emitted by
    scripts/fetch_texeval.py: the glossary document first (every gold term
    one-per-line, so the whole term list resolves to concepts in one
    document), then one Wikipedia-summary document per term. `limit`
    truncates the per-term article documents, never the glossary. Ground
    truth is the gold edge list plus the term universe."""

    def __init__(self, root: str, gold: str, limit: int | None = None):
        self._dir = Path(root) / gold
        self._limit = limit

    def _lines(self, name: str) -> list[str]:
        path = self._dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run `uv run --no-sync python "
                "scripts/fetch_texeval.py` first"
            )
        return path.read_text().splitlines()

    def documents(self) -> Iterator[Document]:
        for i, line in enumerate(self._lines("documents.jsonl")):
            if self._limit is not None and i > self._limit:
                return
            record = json.loads(line)
            yield Document(
                id=record["id"], kind=record["kind"], text=record["text"],
                timestamp=float(i),
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "is_a_edges": [
                json.loads(line) for line in self._lines("gold_edges.jsonl")
            ],
            "terms": [line for line in self._lines("terms.txt") if line],
        }
