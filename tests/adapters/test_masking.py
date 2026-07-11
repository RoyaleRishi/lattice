from lattice.adapters.scorer.masking import mask_document
from tests.helpers import make_mention, make_unit


def test_masks_single_occurrence_one_mask_token_per_word():
    unit = make_unit(id="d:u0", text="deep learning wins")
    mention = make_mention(surface="deep learning", unit_id="d:u0", span=(0, 13))
    assert mask_document([unit], [mention]) == "[MASK] [MASK] wins"


def test_masks_all_occurrences_across_units():
    u0 = make_unit(id="d:u0", text="graphs model graphs")
    u1 = make_unit(id="d:u1", document_id="d", text="we like graphs")
    mentions = [
        make_mention(surface="graphs", unit_id="d:u0", span=(0, 6)),
        make_mention(surface="graphs", unit_id="d:u0", span=(13, 19)),
        make_mention(surface="graphs", unit_id="d:u1", span=(8, 14)),
    ]
    assert mask_document([u0, u1], mentions) == "[MASK] model [MASK]\nwe like [MASK]"


def test_mask_token_count_matches_surface_word_count():
    unit = make_unit(id="d:u0", text="convolutional neural network layers")
    mention = make_mention(
        surface="convolutional neural network", unit_id="d:u0", span=(0, 28)
    )
    assert mask_document([unit], [mention]) == "[MASK] [MASK] [MASK] layers"


def test_no_mentions_reproduces_document_text():
    u0 = make_unit(id="d:u0", text="alpha")
    u1 = make_unit(id="d:u1", document_id="d", text="beta")
    assert mask_document([u0, u1], []) == "alpha\nbeta"


def test_custom_mask_token():
    unit = make_unit(id="d:u0", text="alpha beta")
    mention = make_mention(surface="alpha", unit_id="d:u0", span=(0, 5))
    assert mask_document([unit], [mention], mask_token="<mask>") == "<mask> beta"
