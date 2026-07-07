"""Contract every Dataset adapter must satisfy: documents arrive in stream
order with unique ids, and ground truth is a dict."""

from lattice.ports import Dataset


class DatasetContract:
    def make_dataset(self) -> Dataset:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_yields_at_least_one_document(self):
        assert list(self.make_dataset().documents())

    def test_document_ids_are_unique(self):
        docs = list(self.make_dataset().documents())
        assert len({d.id for d in docs}) == len(docs)

    def test_documents_arrive_in_timestamp_order(self):
        timestamps = [d.timestamp for d in self.make_dataset().documents()]
        assert timestamps == sorted(timestamps)

    def test_ground_truth_is_a_dict(self):
        assert isinstance(self.make_dataset().ground_truth(), dict)
