from lattice.adapters.segmenter.sentence import SentenceSegmenter
from tests.contracts.segmenter_contract import SegmenterContract
from tests.helpers import make_document


class TestSentenceSegmenter(SegmenterContract):
    def make_segmenter(self) -> SentenceSegmenter:
        return SentenceSegmenter()

    def test_splits_on_terminal_punctuation(self):
        doc = make_document(text="Alpha is here. Beta follows! Is gamma third? Delta ends")
        assert [u.text for u in self.make_segmenter().segment(doc)] == [
            "Alpha is here.",
            "Beta follows!",
            "Is gamma third?",
            "Delta ends",
        ]

    def test_units_are_kind_sentence(self):
        doc = make_document(text="One. Two.")
        units = self.make_segmenter().segment(doc)
        assert units and all(u.kind == "sentence" for u in units)

    def test_no_split_without_following_whitespace(self):
        doc = make_document(text="Version 2.5 of the system works.")
        assert len(self.make_segmenter().segment(doc)) == 1
