"""Batched label scoring and residual-stream caching (T-015, T-020).

One forward pass per prompt gives both behavior (log-probabilities of the two candidate labels
at the decision position) and, optionally, residual states at the `query_item` (last subtoken
of the query item) and `decision` (final prompt token) positions, in the convention of
`icl_geometry.models.loading` (states[0] = embeddings, states[l + 1] = block l output).

Padding: right padding, with per-row indexing of the last real token. With a causal model the
pad tokens cannot affect earlier positions (tested in tests/test_scoring.py).

Contextual calibration (Zhao et al. 2021): the same demos with the content-free query "N/A";
its label log-odds are subtracted from the prompt's margin.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from icl_geometry.models.loading import LoadedModel, ResidualRecorder
from icl_geometry.tasks.templates import TEMPLATES
from icl_geometry.tasks.tokenization import span_token_indices

CONTENT_FREE = "N/A"


def label_token_id(tok, label: str) -> int:
    ids = tok.encode(label, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"label {label!r} is not a single token: {ids}")
    return ids[0]


def encode_record(tok, text: str, span: tuple[int, int] | None) -> tuple[list[int], int | None]:
    """Token ids and the index of the query item's last subtoken (None without a span)."""
    if span is None:
        return tok(text, add_special_tokens=False)["input_ids"], None
    inside, ids, _ = span_token_indices(tok, text, tuple(span))
    return ids, inside[-1]


def content_free_text(rec: dict) -> str:
    template = TEMPLATES[rec["template_id"]]
    header = None
    if rec["condition"] == "explicit_rule":
        header = rec["text"][: rec["text"].index("\n\n") + 2]
    text, _ = template.render([(d["item"], d["label"]) for d in rec["demos"]], CONTENT_FREE,
                              header=header)
    return text


def _batches(order: list[int], batch_size: int) -> Iterator[list[int]]:
    for i in range(0, len(order), batch_size):
        yield order[i : i + batch_size]


@torch.no_grad()
def run_forward(
    lm: LoadedModel,
    seqs: list[list[int]],
    query_idx: list[int | None],
    batch_size: int = 16,
    label_ids: list[tuple[int, int]] | None = None,
    positions: tuple[str, ...] = (),
    progress: bool = False,
):
    """Runs all sequences (sorted by length for efficient batching).

    Returns (label_logps [N, 2] or None, label_mass [N] or None,
             states {position: float16 array [N, n_layers + 1, d_model]}).
    """
    n = len(seqs)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    pad = lm.tokenizer.pad_token_id
    logps = np.zeros((n, 2), dtype=np.float64) if label_ids else None
    mass = np.zeros(n, dtype=np.float64) if label_ids else None
    states = {p: np.zeros((n, lm.n_layers + 1, lm.d_model), dtype=np.float16) for p in positions}
    batches = list(_batches(order, batch_size))
    for b in tqdm(batches, disable=not progress, desc="forward"):
        lens = [len(seqs[i]) for i in b]
        width = max(lens)
        ids = torch.full((len(b), width), pad, dtype=torch.long)
        mask = torch.zeros((len(b), width), dtype=torch.long)
        for r, i in enumerate(b):
            ids[r, : lens[r]] = torch.tensor(seqs[i])
            mask[r, : lens[r]] = 1
        with ResidualRecorder(lm.blocks) as rec:
            out = lm.model(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                           output_hidden_states=bool(positions), use_cache=False)
        rows = torch.arange(len(b))
        last = torch.tensor(lens) - 1
        if label_ids is not None:
            lp = out.logits[rows, last].float().log_softmax(-1).cpu()
            for r, i in enumerate(b):
                a, c = label_ids[i]
                logps[i] = (lp[r, a].item(), lp[r, c].item())
                mass[i] = float(lp[r, a].exp() + lp[r, c].exp())
        for p in positions:
            if p == "decision":
                idx = last
            elif p == "query_item":
                idx = torch.tensor([query_idx[i] for i in b])
            else:
                raise ValueError(p)
            emb = out.hidden_states[0][rows, idx]
            layer_states = [emb] + [rec.outputs[j][rows, idx] for j in range(lm.n_layers)]
            st = torch.stack(layer_states, dim=1).float().cpu().numpy().astype(np.float16)
            states[p][b] = st
    return logps, mass, states


def score_records(
    lm: LoadedModel,
    records: list[dict],
    batch_size: int = 16,
    calibrate: bool = True,
    positions: tuple[str, ...] = (),
    progress: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Behavior table (one row per record) and optional cached states per position."""
    tok = lm.tokenizer
    seqs, qidx, labs = [], [], []
    for r in records:
        ids, qi = encode_record(tok, r["text"], r.get("query_span"))
        seqs.append(ids)
        qidx.append(qi)
        labs.append((label_token_id(tok, r["correct_label"]),
                     label_token_id(tok, r["incorrect_label"])))
    logps, mass, states = run_forward(lm, seqs, qidx, batch_size, labs, positions, progress)
    df = pd.DataFrame({
        "prompt_id": [r["prompt_id"] for r in records],
        "model": lm.model_id,
        "logp_correct": logps[:, 0],
        "logp_incorrect": logps[:, 1],
        "label_mass": mass,
    })
    df["margin"] = df.logp_correct - df.logp_incorrect
    df["correct"] = df.margin > 0
    if calibrate:
        cf_seqs = [encode_record(tok, content_free_text(r), None)[0] for r in records]
        cf_logps, cf_mass, _ = run_forward(lm, cf_seqs, [None] * len(records), batch_size, labs,
                                           (), progress)
        df["cf_margin"] = cf_logps[:, 0] - cf_logps[:, 1]
        df["cf_label_mass"] = cf_mass
        df["margin_calibrated"] = df.margin - df.cf_margin
        df["correct_calibrated"] = df.margin_calibrated > 0
    return df, states
