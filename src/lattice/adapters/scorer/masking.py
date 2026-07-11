"""Masked-document construction for MDERank (M2 spec §6.6). Pure stdlib."""

from collections.abc import Sequence

from lattice.core.types import Mention, Unit


def mask_document(
    units: Sequence[Unit], mentions: Sequence[Mention], mask_token: str = "[MASK]"
) -> str:
    """Rebuild the document text (units joined by "\\n") with every given
    mention's span replaced by mask tokens — one per whitespace token of the
    mention's surface, preserving sequence length per MDERank §3 ("the number
    of MASK used for masking is as same as the number of tokens"). Spans are
    character offsets into their unit's text and must not overlap."""
    spans_by_unit: dict[str, list[tuple[int, int, str]]] = {}
    for m in mentions:
        replacement = " ".join([mask_token] * max(len(m.surface.split()), 1))
        spans_by_unit.setdefault(m.unit_id, []).append(
            (m.span[0], m.span[1], replacement)
        )
    masked_units = []
    for unit in units:
        text = unit.text
        for start, end, replacement in sorted(spans_by_unit.get(unit.id, []), reverse=True):
            text = text[:start] + replacement + text[end:]
        masked_units.append(text)
    return "\n".join(masked_units)
