import pytest

from lattice.adapters.extractor.gazetteer import GazetteerExtractor
from tests.contracts.extractor_contract import ExtractorContract
from tests.helpers import make_unit

ROOT = "tests/fixtures/mini_texeval"


class TestGazetteerContract(ExtractorContract):
    def make_extractor(self):
        return GazetteerExtractor(root=ROOT, gold="contract")


def _extract(text: str, terms: list[str], tmp_path):
    gold_dir = tmp_path / "g"
    gold_dir.mkdir()
    (gold_dir / "terms.txt").write_text("\n".join(terms) + "\n")
    extractor = GazetteerExtractor(root=str(tmp_path), gold="g")
    return extractor.extract([make_unit(id="d:u0", text=text)])


def test_longest_match_wins(tmp_path):
    mentions = _extract("olive oil is nice", ["oil", "olive oil"], tmp_path)
    assert [m.surface for m in mentions] == ["olive oil"]
    assert mentions[0].span == (0, 9)


def test_case_insensitive_and_surface_keeps_original_case(tmp_path):
    [mention] = _extract("Olive Oil!", ["olive oil"], tmp_path)
    assert mention.surface == "Olive Oil"
    assert mention.span == (0, 9)


def test_whole_word_boundaries(tmp_path):
    assert _extract("pineapples and oils", ["apple", "oil"], tmp_path) == []


def test_hyphen_neighbors_do_not_match(tmp_path):
    # "(?<!\\w)/(?!\\w)" treats "-" as a boundary; matching inside a
    # hyphenated compound is allowed, matching inside a word is not.
    [mention] = _extract("olive-oil blend", ["oil"], tmp_path)
    assert mention.span == (6, 9)


def test_matches_after_punctuation(tmp_path):
    [mention] = _extract("we love oil.", ["oil"], tmp_path)
    assert mention.span == (8, 11)


def test_multiple_occurrences_all_reported(tmp_path):
    mentions = _extract("oil, oil and oil", ["oil"], tmp_path)
    assert [m.span for m in mentions] == [(0, 3), (5, 8), (13, 16)]


def test_context_windows_the_match(tmp_path):
    text = "x" * 100 + " oil " + "y" * 100
    [mention] = _extract(text, ["oil"], tmp_path)
    start, end = mention.span
    assert mention.context == text[start - 40 : end + 40]


def test_missing_terms_file_names_the_fetch_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_texeval"):
        GazetteerExtractor(root=str(tmp_path), gold="absent")
