"""Residual-stream extraction for arbitrary texts (T-020).

Convention (same as `icl_geometry.models.loading` and `jlens`): states[0] is the token
embedding; states[l + 1] is the output of decoder block l, captured with a forward hook. HF's
last `hidden_states` entry is *not* used for the final layer because it has the final norm
applied; the hook gives the pre-norm block output that `jlens` transports. Shape per position:
[n_layers + 1, d_model], stored as float16.

Positions are located with tokenizer offset mappings from character spans recorded when the
text was built (`Template.render`), so they're robust to multi-token items and right padding.
"""

from __future__ import annotations

import numpy as np

from icl_geometry.behavior.scoring import encode_record, run_forward
from icl_geometry.models.loading import LoadedModel


def extract_states(
    lm: LoadedModel,
    texts: list[str],
    spans: list[tuple[int, int]],
    positions: tuple[str, ...] = ("query_item",),
    batch_size: int = 16,
    progress: bool = False,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """States at the last subtoken of each span (`query_item`) and/or the final token
    (`decision`). Returns ({position: [N, n_layers + 1, d_model] float16}, item token index)."""
    seqs, qidx = [], []
    for text, span in zip(texts, spans, strict=True):
        ids, qi = encode_record(lm.tokenizer, text, span)
        seqs.append(ids)
        qidx.append(qi)
    _, _, states = run_forward(lm, seqs, qidx, batch_size, None, positions, progress)
    return states, qidx
