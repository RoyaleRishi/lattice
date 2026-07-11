import pytest

from lattice.adapters.dataset.mention_clusters import MentionClustersDataset
from tests.contracts.dataset_contract import DatasetContract

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"
CONEL_ROOT = "tests/fixtures/mini_clusters_conel"


class TestMentionClustersDataset(DatasetContract):
    def make_dataset(self) -> MentionClustersDataset:
        return MentionClustersDataset(root=ECB_ROOT)

    def test_documents_carry_stored_kind_and_text(self):
        docs = list(MentionClustersDataset(root=CONEL_ROOT).documents())
        assert [d.kind for d in docs] == ["transcript"] * 3
        assert docs[0].text.startswith("My favorite singer")

    def test_ground_truth_keys_and_clusters(self):
        truth = MentionClustersDataset(root=ECB_ROOT).ground_truth()
        assert truth == {
            "clusters_by_mention": {
                "36_1ecbplus:0-12": "HUM1",
                "36_1ecbplus:33-44": "LOC1",
                "36_2ecbplus:0-5": "HUM1",
                "36_3ecbplus:2-6": "36_3ecbplus:m1",
            }
        }

    def test_spans_slice_stored_text(self):
        for root in (ECB_ROOT, CONEL_ROOT):
            import json
            from pathlib import Path

            for line in (Path(root) / "test.jsonl").read_text().splitlines():
                record = json.loads(line)
                for m in record["mentions"]:
                    assert record["text"][m["start"]:m["end"]] == m["surface"]

    def test_limit_truncates(self):
        docs = list(MentionClustersDataset(root=ECB_ROOT, limit=1).documents())
        assert len(docs) == 1

    def test_missing_root_names_fetch_scripts(self):
        with pytest.raises(FileNotFoundError, match="fetch_ecbplus|fetch_conel2"):
            list(MentionClustersDataset(root="data/nowhere").documents())
