from lattice.adapters.dataset.toy import ToyDataset
from tests.contracts.dataset_contract import DatasetContract


class TestToyDataset(DatasetContract):
    def make_dataset(self) -> ToyDataset:
        return ToyDataset()

    def test_has_three_documents(self):
        assert len(list(ToyDataset().documents())) == 3

    def test_ground_truth_lists_expected_labels(self):
        truth = ToyDataset().ground_truth()
        assert truth["concept_labels"] == ["vector", "store", "embeddings", "encoder"]
