"""Contract every Segmenter adapter must satisfy (spec §11: LSP backbone).
Subclass in the adapter's test module and implement make_segmenter()."""

from lattice.ports import Segmenter
from tests.helpers import make_document


class SegmenterContract:
    def make_segmenter(self) -> Segmenter:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_units_reference_source_document(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert units, "expected at least one unit for non-empty text"
        assert all(u.document_id == doc.id for u in units)

    def test_units_are_ordered_from_zero(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert [u.order for u in units] == list(range(len(units)))

    def test_unit_ids_are_unique(self):
        doc = make_document(text="First block.\n\nSecond block.")
        units = self.make_segmenter().segment(doc)
        assert len({u.id for u in units}) == len(units)

    def test_empty_text_yields_no_units(self):
        assert self.make_segmenter().segment(make_document(text="")) == []
