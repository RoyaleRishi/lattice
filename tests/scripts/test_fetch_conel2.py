import pytest
from scripts.fetch_conel2 import convert_dialogue

SAMPLE = {
    "dialogue_id": "42",
    "turns": [
        {
            "turn_number": 0,
            "speaker": "USER",
            "utterance": "I love the Beatles. so much",
            # span [11, 19] slices "Beatles." — the one known corpus defect;
            # the correction rule must trim it to the mention.
            "el_annotations": [
                {"mention": "Beatles", "span": [11, 19], "entity": "The_Beatles"}
            ],
            "personal_entity_annotations": [
                {"personal_entity_mention": "ignored", "entity": "Ignored"}
            ],
        },
        # Trailing whitespace (37 of 290 raw conversations have it) must be
        # rstripped or BlockSegmenter's strip() breaks the single-unit
        # invariant; a whitespace-only turn must vanish entirely or the
        # joined text gains a blank line.
        {"turn_number": 1, "speaker": "SYSTEM", "utterance": "Me too! "},
        {"turn_number": 2, "speaker": "SYSTEM", "utterance": "   "},
        {
            "turn_number": 3,
            "speaker": "USER",
            "utterance": "kid rock is fine  ",
            "el_annotations": [
                {"mention": "kid rock", "span": [0, 8], "entity": "Kid_Rock"}
            ],
            "personal_entity_annotations": [],
        },
    ],
}


def test_convert_dialogue_remaps_spans_and_corrects_the_known_defect():
    row = convert_dialogue(SAMPLE)
    assert row["id"] == "conel-42"
    assert row["kind"] == "transcript"
    assert row["text"] == "I love the Beatles. so much\nMe too!\nkid rock is fine"
    assert row["mentions"] == [
        {"start": 11, "end": 18, "surface": "Beatles", "cluster": "The_Beatles"},
        {"start": 36, "end": 44, "surface": "kid rock", "cluster": "Kid_Rock"},
    ]
    for m in row["mentions"]:
        assert row["text"][m["start"]:m["end"]] == m["surface"]


def test_emitted_text_survives_the_block_segmenter():
    text = convert_dialogue(SAMPLE)["text"]
    assert text == text.strip()
    assert "\n\n" not in text


def test_personal_entity_annotations_are_excluded():
    row = convert_dialogue(SAMPLE)
    assert all(m["cluster"] != "Ignored" for m in row["mentions"])


def test_duplicate_span_same_cluster_is_deduped():
    dialogue = {
        "dialogue_id": "7",
        "turns": [
            {
                "turn_number": 0,
                "speaker": "USER",
                "utterance": "kid rock rocks",
                "el_annotations": [
                    {"mention": "kid rock", "span": [0, 8], "entity": "Kid_Rock"},
                    {"mention": "kid rock", "span": [0, 8], "entity": "Kid_Rock"},
                ],
            }
        ],
    }
    row = convert_dialogue(dialogue)
    assert len(row["mentions"]) == 1


def test_duplicate_span_conflicting_clusters_raise():
    dialogue = {
        "dialogue_id": "8",
        "turns": [
            {
                "turn_number": 0,
                "speaker": "USER",
                "utterance": "kid rock rocks",
                "el_annotations": [
                    {"mention": "kid rock", "span": [0, 8], "entity": "Kid_Rock"},
                    {"mention": "kid rock", "span": [0, 8], "entity": "Dude"},
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="conflicting clusters"):
        convert_dialogue(dialogue)


def test_unfixable_span_raises():
    broken = {
        "dialogue_id": "9",
        "turns": [
            {
                "turn_number": 0,
                "speaker": "USER",
                "utterance": "hello world",
                "el_annotations": [
                    {"mention": "zzz", "span": [0, 3], "entity": "Zzz"}
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="dialogue 9"):
        convert_dialogue(broken)
