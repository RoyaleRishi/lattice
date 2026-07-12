from scripts.fetch_texeval import (
    GOLDS,
    build_documents,
    normalize_terms,
    parse_taxo,
    recall_ceiling,
    slugify,
    usable_extract,
)


def test_normalize_terms_lowercases_and_dedupes_preserving_order():
    lines = ["0\tAdriatic Sea", "1\tolive oil", "2\tADRIATIC SEA", "", "3\tfat"]
    assert normalize_terms(lines) == ["adriatic sea", "olive oil", "fat"]


def test_parse_taxo_lowercases_and_dedupes():
    lines = [
        "0\tChocos\tbreakfast cereal",
        "1\tchocos\tBREAKFAST CEREAL",
        "",
        "2\twaffle crisp\tbreakfast cereal",
    ]
    assert parse_taxo(lines) == [
        ["chocos", "breakfast cereal"],
        ["waffle crisp", "breakfast cereal"],
    ]


def test_usable_extract_requires_standard_type_and_text():
    assert usable_extract({"type": "standard", "extract": " Olive oil is… "}) == (
        "Olive oil is…"
    )
    assert usable_extract({"type": "disambiguation", "extract": "x"}) is None
    assert usable_extract({"type": "not-found"}) is None
    assert usable_extract({"type": "standard", "extract": "  "}) is None
    assert usable_extract({"type": "standard"}) is None


def test_slugify():
    assert slugify("olive oil") == "olive-oil"
    assert slugify("fisherman's soup") == "fishermans-soup"
    assert slugify("pulp/paper technology") == "pulppaper-technology"
    assert slugify("st. louis-style pizza") == "st-louis-style-pizza"


def test_build_documents_glossary_first_then_articles_in_term_order():
    docs = build_documents(
        "toy",
        ["oil", "olive oil", "fat"],
        {"olive oil": "Olive oil is a fat.", "oil": "Oil is a liquid."},
    )
    assert [d["id"] for d in docs] == ["toy:glossary", "toy:oil", "toy:olive-oil"]
    assert docs[0] == {
        "id": "toy:glossary",
        "kind": "terminology",
        "text": "oil\nolive oil\nfat",
    }
    assert docs[2] == {
        "id": "toy:olive-oil",
        "kind": "article",
        "term": "olive oil",
        "text": "Olive oil is a fat.",
    }


def test_build_documents_slug_collisions_get_numeric_suffixes():
    docs = build_documents(
        "toy",
        ["a b", "a-b"],
        {"a b": "one", "a-b": "two"},
    )
    assert [d["id"] for d in docs] == ["toy:glossary", "toy:a-b", "toy:a-b-2"]


def test_recall_ceiling_counts_edges_with_both_endpoints_in_terms():
    terms = ["a", "b", "c"]
    edges = [["a", "b"], ["a", "z"], ["z", "b"], ["c", "a"]]
    assert recall_ceiling(terms, edges) == (2, 4)


def test_gold_keys_are_the_six_configured_english_golds():
    assert list(GOLDS) == [
        "env-eurovoc",
        "food",
        "food-wordnet",
        "science",
        "science-eurovoc",
        "science-wordnet",
    ]
