"""Zero-shot knowledge verification (T-013).

Forced-choice questions scored by the summed log-probability of each answer continuation
(answers may be multi-token, e.g. " mammal"). Format:

    <question>\\nAnswer:<answer>

Exp 2: per item x attribute, the domain's two phrasings (`questions` in configs/domains.yaml;
they list the options in opposite orders). Exp 1: per item, the true category vs `n_foils`
randomly drawn other categories, two phrasings x both option orders.

Pass rule (D-014, proposed): strict, i.e. the true answer wins in every phrasing and order.
Margins are stored so a threshold rule can be applied later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from icl_geometry.models.loading import LoadedModel

ANSWER_PREFIX = "\nAnswer:"


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _fill(q: str, item: str, a: str = "", b: str = "") -> str:
    text = q.format(item=item, a=a, b=b)
    # fix "a" before vowels for the generic templates ("a kind of animal" is fine; "a apple")
    for w in (item, a, b):
        if w:
            text = text.replace(f" a {w}", f" {article(w)} {w}")
    return text


def exp2_questions(domains: dict, domain: str, items: pd.DataFrame) -> list[dict]:
    spec = domains["exp2"][domain]
    rows = []
    for attr, aspec in spec["attributes"].items():
        if "questions" not in aspec:
            continue
        values = [str(v) for v in aspec["values"]]
        for _, r in items.iterrows():
            true = str(r[attr])
            foil = values[1 - values.index(true)]
            for k, q in enumerate(aspec["questions"]):
                rows.append({
                    "bank": domain, "item": r["item"], "attribute": attr, "true": true,
                    "foil": foil, "phrasing": k, "order": 0,
                    "text": _fill(q, r["item"]) + ANSWER_PREFIX,
                    "answer_true": aspec["answers"][true], "answer_foil": aspec["answers"][foil],
                })
    return rows


def exp1_questions(domains: dict, items: pd.DataFrame, n_foils: int = 3,
                   seed: int = 0) -> list[dict]:
    names = domains["exp1"].get("category_names", {})
    cats = sorted(items.category.unique())
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in items.iterrows():
        others = [c for c in cats if c != r["category"]]
        for foil in rng.choice(others, size=min(n_foils, len(others)), replace=False):
            t, f = names.get(r["category"], r["category"]), names.get(foil, foil)
            for k, q in enumerate(domains["exp1"]["questions"]):
                for order, (a, b) in enumerate(((t, f), (f, t))):
                    rows.append({
                        "bank": "categories", "item": r["item"], "attribute": "category",
                        "true": r["category"], "foil": str(foil), "phrasing": k, "order": order,
                        "text": _fill(q, r["item"], a, b) + ANSWER_PREFIX,
                        "answer_true": f" {article(t)} {t}", "answer_foil": f" {article(f)} {f}",
                    })
    return rows


@torch.no_grad()
def continuation_logprobs(lm: LoadedModel, prompts: list[str], conts: list[str],
                          batch_size: int = 32, progress: bool = False) -> np.ndarray:
    """Summed log p(continuation | prompt) for each pair (right padding)."""
    tok = lm.tokenizer
    seqs, starts = [], []
    for p, c in zip(prompts, conts, strict=True):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ci = tok(p + c, add_special_tokens=False)["input_ids"]
        if ci[: len(pi)] != pi:  # tokenization merged across the boundary; fall back
            ci = pi + tok(c, add_special_tokens=False)["input_ids"]
        seqs.append(ci)
        starts.append(len(pi))
    out = np.zeros(len(seqs))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for s in tqdm(range(0, len(order), batch_size), disable=not progress, desc="knowledge"):
        b = order[s : s + batch_size]
        width = max(len(seqs[i]) for i in b)
        ids = torch.full((len(b), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, i in enumerate(b):
            ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
            mask[r, : len(seqs[i])] = 1
        logits = lm.model(input_ids=ids.to(lm.device), attention_mask=mask.to(lm.device),
                          use_cache=False).logits.float().log_softmax(-1).cpu()
        for r, i in enumerate(b):
            tgt = torch.tensor(seqs[i][starts[i]:])
            pos = torch.arange(starts[i] - 1, len(seqs[i]) - 1)
            out[i] = logits[r, pos, tgt].sum().item()
    return out


def score_questions(lm: LoadedModel, rows: list[dict], batch_size: int = 32,
                    progress: bool = False) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    prompts = list(df.text) * 2
    conts = list(df.answer_true) + list(df.answer_foil)
    lp = continuation_logprobs(lm, prompts, conts, batch_size, progress)
    n = len(df)
    df["logp_true"], df["logp_foil"] = lp[:n], lp[n:]
    df["margin"] = df.logp_true - df.logp_foil
    df["correct"] = df.margin > 0
    return df


def validity_table(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (bank, item, attribute): strict pass, mean/min margin, n questions."""
    return (
        scored.groupby(["bank", "item", "attribute"], sort=False)
        .agg(pass_strict=("correct", "all"), frac_correct=("correct", "mean"),
             mean_margin=("margin", "mean"), min_margin=("margin", "min"),
             n_questions=("correct", "size"))
        .reset_index()
    )
