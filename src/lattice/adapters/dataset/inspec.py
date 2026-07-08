import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "inspec")
class InspecDataset(Dataset):
    """Inspec keyphrase benchmark (Hulth 2003) read from the plain JSONL
    emitted by scripts/fetch_datasets.py. Gold = the uncontrolled keyword
    set as merged by the midas/inspec distribution (M2 spec §6.4).
    Stdlib-only at runtime."""

    def __init__(self, root: str = "data/inspec", split: str = "test", limit: int | None = None):
        self.path = Path(root) / f"{split}.jsonl"
        self.limit = limit

    def _records(self) -> Iterator[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run `uv run --group ml python "
                f"scripts/fetch_datasets.py inspec` first"
            )
        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    return
                yield json.loads(line)

    def documents(self) -> Iterator[Document]:
        for i, record in enumerate(self._records()):
            yield Document(
                id=record["id"], kind="abstract", text=record["text"], timestamp=float(i)
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "keyphrases_by_document": {
                record["id"]: list(record["keyphrases"]) for record in self._records()
            }
        }
