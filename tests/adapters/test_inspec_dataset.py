import pytest

from lattice.adapters.dataset.inspec import InspecDataset
from tests.contracts.dataset_contract import DatasetContract

FIXTURE_ROOT = "tests/fixtures/mini_inspec"


class TestInspecDataset(DatasetContract):
    def make_dataset(self) -> InspecDataset:
        return InspecDataset(root=FIXTURE_ROOT, split="test")

    def test_documents_have_abstract_kind_and_text(self):
        docs = list(self.make_dataset().documents())
        assert len(docs) == 3
        assert docs[0].kind == "abstract"
        assert "Vector databases" in docs[0].text

    def test_ground_truth_keyed_by_document(self):
        truth = self.make_dataset().ground_truth()
        assert truth["keyphrases_by_document"]["mini-2"] == [
            "sentence embeddings", "semantic similarity",
        ]

    def test_limit_truncates(self):
        assert len(list(InspecDataset(root=FIXTURE_ROOT, split="test", limit=2).documents())) == 2

    def test_missing_file_names_the_fetch_script(self):
        with pytest.raises(FileNotFoundError, match="fetch_datasets"):
            list(InspecDataset(root="does/not/exist").documents())
