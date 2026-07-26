from scripts.fetch_multiwoz import convert_dialogue, select_dialogues

SAMPLE = {
    "dialogue_id": "MUL0484.json",
    "services": ["train"],
    "turns": [
        {"speaker": "USER", "turn_id": "0",
         "utterance": "I need a train from norwich to cambridge"},
        {"speaker": "SYSTEM", "turn_id": "1", "utterance": "What time? "},
        {"speaker": "USER", "turn_id": "2", "utterance": "   "},
        {"speaker": "USER", "turn_id": "3", "utterance": "After 10:00  "},
    ],
}


def test_convert_dialogue_joins_turns_and_strips_id_suffix():
    row = convert_dialogue(SAMPLE)
    assert row["id"] == "multiwoz-MUL0484"
    assert row["kind"] == "transcript"
    assert row["text"] == (
        "I need a train from norwich to cambridge\nWhat time?\nAfter 10:00"
    )
    assert row["mentions"] == []


def test_convert_dialogue_text_survives_block_segmenter():
    # BlockSegmenter strips and splits on blank lines; a trailing space or a
    # whitespace-only turn would break the emitted text. rstrip + drop-blank.
    text = convert_dialogue(SAMPLE)["text"]
    assert text == text.strip()
    assert "\n\n" not in text


def test_convert_dialogue_single_turn():
    row = convert_dialogue(
        {"dialogue_id": "SNG01.json",
         "turns": [{"speaker": "USER", "turn_id": "0", "utterance": "hello"}]}
    )
    assert row["text"] == "hello"
    assert row["id"] == "multiwoz-SNG01"


def test_select_dialogues_is_deterministic_and_sorted():
    # 15 dialogues across three id prefixes; sorted-first-4 must be reproducible
    # and independent of input order. 'M' < 'P' < 'S', so MUL* come first.
    dialogues = [
        {"dialogue_id": f"{p}{i}.json", "turns": []}
        for p in ("SNG", "MUL", "PMUL")
        for i in range(5)
    ]
    selected = select_dialogues(dialogues, limit=4)
    ids = [d["dialogue_id"] for d in selected]
    assert ids == sorted(ids)
    assert len(selected) == 4
    assert all(i.startswith("MUL") for i in ids)
