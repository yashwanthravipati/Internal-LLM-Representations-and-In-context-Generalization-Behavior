"""Tokenization checks for items, labels and attribute words (T-012).

Items are tokenized in context: the query item rendered by a prompt template after one demo
line. Character offsets locate the item's tokens, so counts reflect exactly what the model sees.
"""

from __future__ import annotations

from dataclasses import dataclass

from icl_geometry.tasks.templates import TEMPLATES, Template

REFERENCE_DEMO = [("dog", " foo")]


@dataclass
class ItemTokens:
    item: str
    n_tokens: int
    token_ids: list[int]
    clean_boundary: bool  # item span starts and ends on token boundaries
    suffix_ok: bool  # tokens after the item match those after a reference item


def span_token_indices(tok, text: str, span: tuple[int, int]) -> tuple[list[int], list[int], bool]:
    """(token indices overlapping span, all input ids, clean boundary)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    start, end = span
    inside = [i for i, (s, e) in enumerate(offsets) if s < end and e > start]
    # The first token may include the item's leading space (" dog"); that still counts as clean.
    first_start = offsets[inside[0]][0] if inside else -1
    starts_ok = first_start == start or (first_start == start - 1 and text[start - 1] == " ")
    clean = bool(inside) and starts_ok and offsets[inside[-1]][1] == end
    return inside, ids, clean


def _suffix_ids(tok, template: Template, item: str) -> tuple[list[int], list[int], bool]:
    text, span = template.render(REFERENCE_DEMO, item)
    inside, ids, clean = span_token_indices(tok, text, span)
    item_ids = [ids[i] for i in inside]
    suffix = ids[inside[-1] + 1 :] if inside else []
    return item_ids, suffix, clean


def item_tokens(tok, item: str, template: str | Template = "arrow") -> ItemTokens:
    t = TEMPLATES[template] if isinstance(template, str) else template
    item_ids, suffix, clean = _suffix_ids(tok, t, item)
    _, ref_suffix, _ = _suffix_ids(tok, t, "cat")
    return ItemTokens(item, len(item_ids), item_ids, clean, suffix == ref_suffix)


def single_token_id(tok, word: str, leading_space: bool = True) -> int | None:
    """Token id of `word` (with a leading space by default) if it is a single token."""
    ids = tok.encode((" " if leading_space else "") + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None
