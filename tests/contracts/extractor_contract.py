"""Contract every Extractor adapter must satisfy. Spans must index into the
owning unit's text, and the spanned slice must match the surface
case-insensitively (adapters may normalize surface case)."""

from lattice.ports import Extractor
from tests.helpers import make_unit


class ExtractorContract:
    def make_extractor(self) -> Extractor:
        raise NotImplementedError("subclass must provide the adapter under test")

    def test_mentions_reference_their_units(self):
        units = [
            make_unit(id="d:u0", text="Vector stores index embeddings."),
            make_unit(id="d:u1", text="Encoders produce embeddings.", order=1),
        ]
        mentions = self.make_extractor().extract(units)
        assert mentions, "expected mentions from non-trivial text"
        unit_ids = {u.id for u in units}
        assert all(m.unit_id in unit_ids for m in mentions)

    def test_spans_slice_the_unit_text(self):
        units = [make_unit(id="d:u0", text="Vector stores index embeddings.")]
        mentions = self.make_extractor().extract(units)
        by_id = {u.id: u for u in units}
        for m in mentions:
            start, end = m.span
            assert 0 <= start < end <= len(by_id[m.unit_id].text)
            assert by_id[m.unit_id].text[start:end].lower() == m.surface.lower()

    def test_no_units_yields_no_mentions(self):
        assert self.make_extractor().extract([]) == []
