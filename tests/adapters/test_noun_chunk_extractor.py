import pytest

pytestmark = pytest.mark.ml
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip(
        "en_core_web_sm not installed (run scripts/fetch_models.py)",
        allow_module_level=True,
    )

from lattice.adapters.extractor.noun_chunk import NounChunkExtractor  # noqa: E402
from tests.contracts.extractor_contract import ExtractorContract  # noqa: E402
from tests.helpers import make_unit  # noqa: E402


class TestNounChunkExtractor(ExtractorContract):
    def make_extractor(self) -> NounChunkExtractor:
        return NounChunkExtractor()

    def test_extracts_noun_phrases_not_verbs(self):
        mentions = self.make_extractor().extract(
            [make_unit(text="The vector store indexes dense embeddings.")]
        )
        surfaces = {m.surface for m in mentions}
        assert "vector store" in surfaces
        assert "dense embeddings" in surfaces
        assert not any("indexes" == s for s in surfaces)

    def test_leading_determiner_trimmed(self):
        mentions = self.make_extractor().extract([make_unit(text="The encoder produces vectors.")])
        assert "encoder" in {m.surface for m in mentions}
        assert not any(s.startswith("the ") for s in {m.surface for m in mentions})

    def test_pronoun_only_chunks_dropped(self):
        mentions = self.make_extractor().extract([make_unit(text="It maps text to vectors.")])
        assert "it" not in {m.surface for m in mentions}

    def test_max_tokens_filters_long_chunks(self):
        text = "The very long deeply nested compound noun phrase construction persists."
        short = NounChunkExtractor(max_tokens=2).extract([make_unit(text=text)])
        assert all(len(m.surface.split()) <= 2 for m in short)
