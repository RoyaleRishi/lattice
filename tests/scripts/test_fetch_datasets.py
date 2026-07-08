import json

from scripts.fetch_datasets import record_to_line


def test_record_to_line_joins_tokens_and_merges_keyphrases():
    record = {
        "id": "1789",
        "document": ["Neural", "networks", "learn", "representations", "."],
        "extractive_keyphrases": ["neural networks"],
        "abstractive_keyphrases": ["representation learning"],
    }
    parsed = json.loads(record_to_line(record))
    assert parsed == {
        "id": "1789",
        "text": "Neural networks learn representations .",
        "keyphrases": ["neural networks", "representation learning"],
    }


def test_record_to_line_missing_id_uses_fallback():
    record = {"document": ["x"], "extractive_keyphrases": [], "abstractive_keyphrases": []}
    parsed = json.loads(record_to_line(record, fallback_id="42"))
    assert parsed["id"] == "42"
