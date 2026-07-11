from scripts.fetch_ecbplus import convert_document, is_entity_tag

SAMPLE_XML = """<Document doc_name="1_1ecbplus.xml" doc_id="DOC1">
<token t_id="1" sentence="0" number="0">http</token>
<token t_id="2" sentence="1" number="0">Warren</token>
<token t_id="3" sentence="1" number="1">Jeffs</token>
<token t_id="4" sentence="1" number="2">guilty</token>
<token t_id="5" sentence="2" number="0">Jury</token>
<token t_id="6" sentence="2" number="1">decides</token>
<Markables>
<HUMAN_PART_PER m_id="1"><token_anchor t_id="2"/><token_anchor t_id="3"/></HUMAN_PART_PER>
<ACTION_OCCURRENCE m_id="2"><token_anchor t_id="4"/></ACTION_OCCURRENCE>
<HUMAN_PART_ORG m_id="3"><token_anchor t_id="5"/></HUMAN_PART_ORG>
<HUMAN_PART_PER m_id="4"><token_anchor t_id="1"/></HUMAN_PART_PER>
<HUMAN_PART_PER m_id="9" RELATED_TO="" TAG_DESCRIPTOR="jeffs" instance_id="HUM99"/>
</Markables>
<Relations>
<CROSS_DOC_COREF r_id="10" note="HUM99"><source m_id="1"/><target m_id="9"/></CROSS_DOC_COREF>
</Relations>
</Document>"""


def test_entity_tag_rule():
    assert is_entity_tag("HUMAN_PART_PER")
    assert is_entity_tag("NON_HUMAN_PART")
    assert is_entity_tag("LOC_GEO")
    assert is_entity_tag("TIME_DATE")
    assert not is_entity_tag("ACTION_OCCURRENCE")
    assert not is_entity_tag("NEG_ACTION_STATE")
    assert not is_entity_tag("UNKNOWN_INSTANCE_TAG")


def test_convert_document_builds_text_spans_and_clusters():
    row = convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences={"1", "2"})
    assert row["id"] == "1_1ecbplus"
    assert row["kind"] == "article"
    # sentence 0 (the URL junk) is excluded by the validated filter; tokens
    # joined with spaces, sentences with newline.
    assert row["text"] == "Warren Jeffs guilty\nJury decides"
    assert row["mentions"] == [
        {"start": 0, "end": 12, "surface": "Warren Jeffs", "cluster": "HUM99"},
        {"start": 20, "end": 24, "surface": "Jury", "cluster": "1_1ecbplus:m3"},
    ]
    for m in row["mentions"]:
        assert row["text"][m["start"]:m["end"]] == m["surface"]


def test_action_markables_and_unvalidated_sentences_are_excluded():
    row = convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences={"1", "2"})
    surfaces = [m["surface"] for m in row["mentions"]]
    assert "guilty" not in surfaces  # ACTION_OCCURRENCE
    assert "http" not in surfaces  # sentence 0 not validated


def test_no_validated_sentences_returns_none():
    assert convert_document("1_1ecbplus", SAMPLE_XML, validated_sentences=set()) is None
