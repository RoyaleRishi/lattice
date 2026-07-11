import pytest

from lattice.adapters.extractor.gold_mentions import GoldMentionExtractor
from lattice.adapters.segmenter.block import BlockSegmenter
from lattice.core.types import Document
from tests.helpers import make_unit

ECB_ROOT = "tests/fixtures/mini_clusters_ecb"


def _units_for(doc_id: str, text: str):
    return BlockSegmenter().segment(
        Document(id=doc_id, kind="article", text=text, timestamp=0.0)
    )


class TestGoldMentionExtractor:
    def make_extractor(self) -> GoldMentionExtractor:
        return GoldMentionExtractor(root=ECB_ROOT)

    def test_emits_gold_mentions_with_valid_spans(self):
        text = "Warren Jeffs was found guilty in San Antonio ."
        units = _units_for("36_1ecbplus", text)
        assert len(units) == 1  # single-unit invariant (spec §4.2)
        mentions = self.make_extractor().extract(units)
        assert [(m.surface, m.span) for m in mentions] == [
            ("Warren Jeffs", (0, 12)),
            ("San Antonio", (33, 44)),
        ]
        assert all(m.unit_id == units[0].id for m in mentions)
        for m in mentions:
            assert units[0].text[m.span[0]:m.span[1]] == m.surface

    def test_no_units_yields_no_mentions(self):
        assert self.make_extractor().extract([]) == []

    def test_unknown_document_raises(self):
        with pytest.raises(ValueError, match="not in gold mention sidecar"):
            self.make_extractor().extract([make_unit(id="x:u0", document_id="unknown-doc")])

    def test_text_mismatch_raises(self):
        units = _units_for("36_1ecbplus", "Tampered text .")
        with pytest.raises(ValueError, match="differs from stored"):
            self.make_extractor().extract(units)

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="fetch"):
            GoldMentionExtractor(root="data/nowhere")
