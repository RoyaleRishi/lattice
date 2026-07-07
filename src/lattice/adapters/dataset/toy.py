from collections.abc import Iterator

from lattice.core.types import Document
from lattice.ports import Dataset
from lattice.registry.registry import register


@register(Dataset, "toy")
class ToyDataset(Dataset):
    """Three tiny in-code documents for the walking skeleton. Real benchmark
    datasets (Inspec, SemEval, ECB+) arrive in Milestones 2-4 behind the
    same port."""

    _DOCS = (
        Document(
            id="doc-1",
            kind="note",
            timestamp=1.0,
            text="The vector store indexes embeddings.\n\nThe vector store returns neighbors.",
        ),
        Document(
            id="doc-2",
            kind="note",
            timestamp=2.0,
            text="Embeddings come from the encoder model.",
        ),
        Document(
            id="doc-3",
            kind="note",
            timestamp=3.0,
            text="The vector store holds embeddings from the encoder.",
        ),
    )

    def documents(self) -> Iterator[Document]:
        yield from self._DOCS

    def ground_truth(self) -> dict[str, object]:
        return {"concept_labels": ["vector", "store", "embeddings", "encoder"]}
