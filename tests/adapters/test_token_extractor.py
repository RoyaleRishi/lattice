from lattice.adapters.extractor.token import TokenExtractor
from tests.contracts.extractor_contract import ExtractorContract
from tests.helpers import make_unit


class TestTokenExtractor(ExtractorContract):
    def make_extractor(self) -> TokenExtractor:
        return TokenExtractor()

    def test_short_words_filtered_by_min_length(self):
        mentions = TokenExtractor(min_length=5).extract(
            [make_unit(text="The vector store maps text")]
        )
        assert {m.surface for m in mentions} == {"vector", "store"}

    def test_surfaces_are_lowercased(self):
        mentions = TokenExtractor().extract([make_unit(text="Vector Embeddings")])
        assert {m.surface for m in mentions} == {"vector", "embeddings"}

    def test_every_occurrence_is_a_mention(self):
        mentions = TokenExtractor().extract([make_unit(text="store the store")])
        assert [m.surface for m in mentions] == ["store", "store"]
        assert mentions[0].span != mentions[1].span
