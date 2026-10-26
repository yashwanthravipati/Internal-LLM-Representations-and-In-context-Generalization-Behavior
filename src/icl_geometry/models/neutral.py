"""Neutral (context-free) item representations (T-021).

Each item is embedded in K carrier texts with no demonstrations, and the state at the item's
last subtoken is read at every layer. No carrier puts the item at position 0 (Qwen has no BOS
token, so the first position acts as an attention sink). Carriers mix the two tokenizations
of an item: after a space (" dog") and at the start of a line ("dog", as in the `arrow`
template).

D-011 (proposed): the neutral representation is the mean over carriers; per-carrier states
are kept for the sensitivity analysis (carriers as extra manifold samples).
"""

from __future__ import annotations

import numpy as np

from icl_geometry.models.activations import extract_states
from icl_geometry.models.loading import LoadedModel

CARRIERS = (
    "The word is {item}",
    "Here is one example: {item}",
    "Item: {item}",
    "Next line:\n{item} ->",  # matches the `arrow` template's line-initial tokenization
)


def carrier_texts(item: str) -> list[tuple[str, tuple[int, int]]]:
    out = []
    for c in CARRIERS:
        pre = c.split("{item}")[0]
        out.append((c.format(item=item), (len(pre), len(pre) + len(item))))
    return out


def neutral_states(lm: LoadedModel, items: list[str], batch_size: int = 32,
                   progress: bool = False) -> np.ndarray:
    """[n_items, n_carriers, n_layers + 1, d_model] float16."""
    texts, spans = [], []
    for it in items:
        for text, span in carrier_texts(it):
            texts.append(text)
            spans.append(span)
    states, _ = extract_states(lm, texts, spans, ("query_item",), batch_size, progress)
    arr = states["query_item"]
    return arr.reshape(len(items), len(CARRIERS), *arr.shape[1:])
