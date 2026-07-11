import json
from collections.abc import Iterator
from pathlib import Path

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "mention-clusters")
class MentionClustersDataset(Dataset):
    """Unified mention-cluster benchmark reader (M3 spec §4.4): one JSONL
    shape serves both ECB+ and ConEL-2, emitted by scripts/fetch_ecbplus.py
    and scripts/fetch_conel2.py. Stdlib-only at runtime. Ground truth maps
    mention keys f"{doc_id}:{start}-{end}" to cluster ids."""

    def __init__(self, root: str, split: str = "test", limit: int | None = None):
        self.path = Path(root) / f"{split}.jsonl"
        self.limit = limit

    def _records(self) -> Iterator[dict]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found — run `uv run --no-sync python "
                f"scripts/fetch_ecbplus.py` or `scripts/fetch_conel2.py` first"
            )
        with self.path.open() as f:
            for i, line in enumerate(f):
                if self.limit is not None and i >= self.limit:
                    return
                yield json.loads(line)

    def documents(self) -> Iterator[Document]:
        for i, record in enumerate(self._records()):
            yield Document(
                id=record["id"], kind=record["kind"], text=record["text"],
                timestamp=float(i),
            )

    def ground_truth(self) -> dict[str, object]:
        return {
            "clusters_by_mention": {
                f"{record['id']}:{m['start']}-{m['end']}": m["cluster"]
                for record in self._records()
                for m in record["mentions"]
            }
        }
