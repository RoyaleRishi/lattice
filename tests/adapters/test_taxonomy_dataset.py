import pytest

from lattice.adapters.dataset.taxonomy import TaxonomyDataset
from tests.contracts.dataset_contract import DatasetContract

ROOT = "tests/fixtures/mini_texeval"


class TestTaxonomyContract(DatasetContract):
    def make_dataset(self):
        return TaxonomyDataset(root=ROOT, gold="toy")


def test_glossary_document_comes_first():
    docs = list(TaxonomyDataset(root=ROOT, gold="toy").documents())
    assert docs[0].id == "toy:glossary"
    assert docs[0].kind == "terminology"
    assert docs[0].text.splitlines() == [
        "oil", "olive oil", "vegetable oil", "sunflower oil", "fat",
    ]
    assert [d.timestamp for d in docs] == [0.0, 1.0, 2.0]


def test_limit_truncates_articles_but_never_the_glossary():
    docs = list(TaxonomyDataset(root=ROOT, gold="toy", limit=1).documents())
    assert [d.id for d in docs] == ["toy:glossary", "toy:olive-oil"]
    assert [d.id for d in TaxonomyDataset(root=ROOT, gold="toy", limit=0).documents()] == [
        "toy:glossary"
    ]


def test_ground_truth_shape():
    truth = TaxonomyDataset(root=ROOT, gold="toy").ground_truth()
    assert truth["terms"] == ["oil", "olive oil", "vegetable oil", "sunflower oil", "fat"]
    assert ["olive oil", "oil"] in truth["is_a_edges"]
    assert len(truth["is_a_edges"]) == 5


def test_missing_data_names_the_fetch_script():
    with pytest.raises(FileNotFoundError, match="fetch_texeval"):
        list(TaxonomyDataset(root=ROOT, gold="absent").documents())
