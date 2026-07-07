from lattice.adapters.segmenter.block import BlockSegmenter
from tests.contracts.segmenter_contract import SegmenterContract
from tests.helpers import make_document


class TestBlockSegmenter(SegmenterContract):
    def make_segmenter(self) -> BlockSegmenter:
        return BlockSegmenter()

    def test_splits_on_blank_lines(self):
        units = self.make_segmenter().segment(
            make_document(text="First block.\n\nSecond block.")
        )
        assert [u.text for u in units] == ["First block.", "Second block."]

    def test_strips_whitespace_and_drops_empty_blocks(self):
        units = self.make_segmenter().segment(
            make_document(text="  First.  \n\n\n\n  Second.  ")
        )
        assert [u.text for u in units] == ["First.", "Second."]
