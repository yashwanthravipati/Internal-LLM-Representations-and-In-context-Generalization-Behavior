"""Rule readout with the Jacobian lens (H4, T-041).

For each cached residual (prompt x position) and lens layer, the full-vocabulary lens logits
are computed and each attribute word gets a **percentile rank** in [0, 1] (1 = top of the
vocabulary). Ranks are comparable across positions and layers, unlike raw logits or concept
dot products (see `jlens_wrapper`).

    rule_readout = mean rank(cued attribute words) - mean rank(uncued attribute words)

Positive values mean the demonstrated grouping is "poised to be verbalized" more than the
competing one. Compare main vs random-label (negative control) vs explicit-rule (positive
control) prompts, and correct vs incorrect trials.
"""

from __future__ import annotations

import numpy as np
import torch

from icl_geometry.lens.jlens_wrapper import Lens, readout, state_index
from icl_geometry.models.loading import LoadedModel


def attribute_word_ids(tok, domains: dict, domain: str) -> dict[str, list[int]]:
    """attribute name -> single-token ids (with leading space) of all its value words."""
    out = {}
    for attr, spec in domains["exp2"][domain]["attributes"].items():
        ids = []
        for words in spec["words"].values():
            for w in words:
                t = tok.encode(" " + w, add_special_tokens=False)
                if len(t) == 1:
                    ids.append(t[0])
        out[attr] = sorted(set(ids))
    return out


@torch.no_grad()
def percentile_ranks(lm: LoadedModel, lens: Lens, states: np.ndarray, layers: list[int],
                     token_ids: list[int], batch: int = 16) -> np.ndarray:
    """states [N, n_states, d] -> ranks [N, len(layers), len(token_ids)] in [0, 1]."""
    n = states.shape[0]
    tid = torch.tensor(token_ids)
    out = np.zeros((n, len(layers), len(token_ids)), dtype=np.float32)
    for j, layer in enumerate(layers):
        for s in range(0, n, batch):
            h = torch.from_numpy(states[s : s + batch, state_index(layer)].astype(np.float32))
            logits = readout(lm, lens, h, layer)  # [b, V]
            sel = logits[:, tid]  # [b, T]
            below = (logits[:, None, :] < sel[:, :, None]).sum(-1)  # [b, T]
            out[s : s + batch, j] = (below.float() / (logits.shape[1] - 1)).numpy()
    return out
