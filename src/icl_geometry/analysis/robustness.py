"""Experiment 2 controls and robustness (T-034; PLAN §10).

Re-evaluates the headline Exp 2 claims on subsets of the prompts (one level of one incidental
variable at a time) and under alternative analysis choices, and classifies each cell:

* Claims (each a number with a null value and a direction):
  - `h2_delta_snr_cued`: mean over (domain, cued attribute, m) groups of
    SNR_cued(main) - SNR_cued(random-label control) at one layer (null 0).
  - `h2_selectivity`: the same for (ΔSNR_cued - ΔSNR_uncued): the shift is specific to the
    cued attribute (null 0).
  - `h3_shift_auroc`: test AUROC of the raw shift score for correctness on incongruent
    main-condition queries (null 0.5). The score's direction is fixed in advance (positive =
    toward the item's cued class), so no model is fitted.
* Variations: levels of label_set, label_map (counterbalancing), order, template_id and m
  that occur in the prompts; alternative neutral representations (D-011: single carriers
  instead of the carrier mean); alternative representation transforms (D-012: drop the top-5
  dimensions by mean |activation|, per-dimension standardization).
* CIs: 95% percentile bootstrap over query items (items are the unit that repeats across
  prompts, as in `analysis.prediction`).

Cell rules (`classify_cell`):
* `n/a`: estimate or CI not finite, or fewer than `min_n` prompts in the subset.
* `holds`: point estimate on the reference side of the null and the CI excludes the null.
* `weakens`: the CI includes the null (whatever the side of the point estimate; the cell text
  shows the estimate).
* `fails`: a significant reversal, i.e. the CI excludes the null on the opposite side.
The reference side is the side of the full-data estimate when its CI excludes the null (so a
significant reversal of the hypothesis is the claim being tested for robustness), else the
hypothesized direction (+1 for all three claims): with a null full-data result, subsets are
checked against the hypothesis, not against the sign of noise.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from icl_geometry.analysis.exp2 import attr_values, shift_feature_from
from icl_geometry.analysis.prediction import _group_bootstrap
from icl_geometry.geometry.shift import snr_partition

VARIATION_COLUMNS = ("label_set", "label_map", "order", "template_id", "m")
CLAIMS = {"h2_delta_snr_cued": 0.0, "h2_selectivity": 0.0, "h3_shift_auroc": 0.5}
MIN_GROUP = 8  # prompts per condition per group, as in analysis.exp2.layer_profile


def classify_cell(estimate: float, ci_low: float, ci_high: float, null: float,
                  ref_sign: int, n: int | None = None, min_n: int = 0) -> str:
    """holds / weakens / fails / n/a (rules in the module docstring)."""
    if n is not None and n < min_n:
        return "n/a"
    if not all(np.isfinite(v) for v in (estimate, ci_low, ci_high)):
        return "n/a"
    if ci_low > null:
        return "holds" if ref_sign > 0 else "fails"
    if ci_high < null:
        return "holds" if ref_sign < 0 else "fails"
    return "weakens"


# ---------------------------------------------------------------------------
# H2: ΔSNR relative to the random-label control
# ---------------------------------------------------------------------------


def _group_delta(X: np.ndarray, meta: pd.DataFrame) -> tuple[float, float] | None:
    """(ΔSNR_cued, ΔSNR_uncued) for one (domain, cued_attr, m) group, rows aligned with X."""
    main = (meta.condition == "main").to_numpy()
    ctrl = (meta.condition == "random_label").to_numpy()
    if main.sum() < MIN_GROUP or ctrl.sum() < MIN_GROUP:
        return None
    yc, yu = attr_values(meta, "cued_attr"), attr_values(meta, "uncued_attr")
    m = float(meta.m.iloc[0])
    out = []
    for y in (yc, yu):
        for rows in (y[main], y[ctrl]):  # two classes with >= 2 samples each
            _, counts = np.unique(rows, return_counts=True)
            if len(counts) != 2 or counts.min() < 2:
                return None
        out.append(snr_partition(X[main], y[main], m) - snr_partition(X[ctrl], y[ctrl], m))
    return out[0], out[1]


def h2_claims(X: np.ndarray, meta: pd.DataFrame,
              group_cols=("domain", "cued_attr", "m")) -> dict[str, float]:
    """Mean over groups of ΔSNR_cued and of (ΔSNR_cued - ΔSNR_uncued). X rows align with
    meta rows (positional); only main and random_label rows are used."""
    cued, sel = [], []
    meta = meta.reset_index(drop=True)
    for _, g in meta.groupby(list(group_cols)):
        d = _group_delta(X[g.index.to_numpy()], g)
        if d is not None:
            cued.append(d[0])
            sel.append(d[0] - d[1])
    nan = float("nan")
    return {"h2_delta_snr_cued": float(np.mean(cued)) if cued else nan,
            "h2_selectivity": float(np.mean(sel)) if sel else nan, "n_groups": len(cued)}


def delta_snr_permutation_null(X: np.ndarray, meta: pd.DataFrame, n_perm: int = 200,
                               seed: int = 0, group_cols=("domain", "cued_attr", "m")
                               ) -> np.ndarray:
    """Null distribution of `h2_claims(...)['h2_delta_snr_cued']`: main / random_label
    condition labels shuffled within each group (keeps group sizes and items)."""
    rng = np.random.default_rng(seed)
    meta = meta.reset_index(drop=True)
    keep = meta.condition.isin(["main", "random_label"]).to_numpy()
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    out = np.empty(n_perm)
    for k in range(n_perm):
        perm = meta.copy()
        for _, g in meta.groupby(list(group_cols)):
            perm.loc[g.index, "condition"] = rng.permutation(g.condition.to_numpy())
        out[k] = h2_claims(X, perm, group_cols)["h2_delta_snr_cued"]
    return out


def h2_bootstrap(X: np.ndarray, meta: pd.DataFrame, n_boot: int = 200, seed: int = 0,
                 group_cols=("domain", "cued_attr", "m")) -> pd.DataFrame:
    """Point estimates and 95% CIs of the H2 claims, resampling query items."""
    meta = meta.reset_index(drop=True)
    point = h2_claims(X, meta, group_cols)
    boots = {k: [] for k in ("h2_delta_snr_cued", "h2_selectivity")}
    for idx in _group_bootstrap(meta.query_item.to_numpy(), n_boot, seed):
        b = h2_claims(X[idx], meta.iloc[idx].reset_index(drop=True), group_cols)
        for k in boots:
            if np.isfinite(b[k]):
                boots[k].append(b[k])
    return _ci_frame(point, boots, n=len(meta))


# ---------------------------------------------------------------------------
# H3: shift score predicts correctness on incongruent queries
# ---------------------------------------------------------------------------


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def h3_bootstrap(df: pd.DataFrame, score: np.ndarray, n_boot: int = 200,
                 seed: int = 0) -> pd.DataFrame:
    """AUROC of `score` for `correct` on main-condition incongruent rows of df (positional
    alignment), with a 95% CI over query items."""
    sel = ((df.condition == "main") & ~df.congruent.astype(bool)).to_numpy()
    sub = df[sel].reset_index(drop=True)
    y, s = sub.correct.astype(int).to_numpy(), np.asarray(score)[sel]
    point = {"h3_shift_auroc": _auroc(y, s)}
    boots = {"h3_shift_auroc": []}
    if len(sub):
        for idx in _group_bootstrap(sub.query_item.to_numpy(), n_boot, seed):
            b = _auroc(y[idx], s[idx])
            if np.isfinite(b):
                boots["h3_shift_auroc"].append(b)
    return _ci_frame(point, boots, n=len(sub))


def _ci_frame(point: dict, boots: dict[str, list], n: int) -> pd.DataFrame:
    rows = []
    for k, vals in boots.items():
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) >= 20 else (np.nan, np.nan))
        rows.append({"claim": k, "estimate": point[k], "ci_low": float(lo),
                     "ci_high": float(hi), "n": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alternative analysis choices (D-011, D-012)
# ---------------------------------------------------------------------------


def transform_variants(X_ref: np.ndarray, k_drop: int = 5) -> dict[str, Callable]:
    """D-012 alternatives, with statistics estimated on `X_ref` (e.g. val main-condition
    reps at the layer): identity (primary), drop the top-k dimensions by mean |activation|,
    per-dimension standardization."""
    X_ref = np.asarray(X_ref, dtype=np.float64)
    top = np.argsort(-np.abs(X_ref).mean(0))[:k_drop]
    keep = np.setdiff1d(np.arange(X_ref.shape[1]), top)
    mu, sd = X_ref.mean(0), X_ref.std(0) + 1e-8
    return {"raw": lambda X: X,
            f"drop_top{k_drop}_dims": lambda X: X[:, keep],
            "standardized": lambda X: (X - mu) / sd}


def neutral_variants(df: pd.DataFrame, neutral, row_of, state: int) -> dict[str, np.ndarray]:
    """D-011 alternatives for the neutral rep of each row's query item at `state`: the carrier
    mean (primary) and each single carrier."""
    rows = np.array([row_of[i] for i in df.query_item])
    per = np.asarray(neutral[rows, :, state], dtype=np.float64)  # [n, carriers, d]
    out = {"carrier_mean": per.mean(1)}
    for c in range(per.shape[1]):
        out[f"carrier_{c}"] = per[:, c]
    return out


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def robustness_table(df: pd.DataFrame, X_h2: np.ndarray, X_h3: np.ndarray,
                     Xn_h3: np.ndarray, fit_mask: np.ndarray, eval_mask: np.ndarray,
                     neutral_alts: dict[str, np.ndarray] | None = None,
                     transforms: dict[str, Callable] | None = None,
                     variation_cols=VARIATION_COLUMNS, n_boot: int = 200, seed: int = 0,
                     min_n: int = 16) -> pd.DataFrame:
    """Long table: one row per (claim, variation, level) with estimate, CI, n and cell.

    df: prompt table (positional rows) with a `cued_value` column. X_h2: in-context reps at
    the H2 layer; X_h3 / Xn_h3: in-context and neutral reps at the H3 layer. Shift axes are fit
    on `fit_mask` rows (val); claims are evaluated on `eval_mask` rows (test). Prompt-subset
    variations restrict evaluation rows only; analysis variations (neutral_alts, transforms)
    recompute the inputs and refit the axes.
    """
    df = df.reset_index(drop=True)
    fit_mask, eval_mask = np.asarray(fit_mask, bool), np.asarray(eval_mask, bool)

    def evaluate(mask, Xa, Xb, Xn):
        score = shift_feature_from(df, Xb, Xn, fit_mask)
        sub = df[mask].reset_index(drop=True)
        h2 = h2_bootstrap(Xa[mask], sub, n_boot, seed)
        h3 = h3_bootstrap(sub, score[mask], n_boot, seed)
        return pd.concat([h2, h3], ignore_index=True)

    frames = []
    full = evaluate(eval_mask, X_h2, X_h3, Xn_h3).assign(variation="all", level="all")
    frames.append(full)
    for col in variation_cols:
        if col not in df or df.loc[eval_mask, col].nunique() < 2:
            continue
        for level in sorted(df.loc[eval_mask, col].unique(), key=str):
            mask = eval_mask & (df[col] == level).to_numpy()
            frames.append(evaluate(mask, X_h2, X_h3, Xn_h3)
                          .assign(variation=col, level=str(level)))
    for name, Xn in (neutral_alts or {}).items():
        r = evaluate(eval_mask, X_h2, X_h3, Xn)
        frames.append(r[r.claim == "h3_shift_auroc"].assign(variation="neutral", level=name))
    for name, f in (transforms or {}).items():
        frames.append(evaluate(eval_mask, f(X_h2), f(X_h3), f(Xn_h3))
                      .assign(variation="transform", level=name))
    out = pd.concat(frames, ignore_index=True)

    ref = {}
    for r in full.itertuples():
        null = CLAIMS[r.claim]
        significant = np.isfinite(r.ci_low) and (r.ci_low > null or r.ci_high < null)
        ref[r.claim] = int(np.sign(r.estimate - null)) if significant else 1
    out["ref_sign"] = out.claim.map(ref)
    out["cell"] = [classify_cell(r.estimate, r.ci_low, r.ci_high, CLAIMS[r.claim], r.ref_sign,
                                 n=r.n, min_n=min_n) for r in out.itertuples()]
    return out[["claim", "variation", "level", "estimate", "ci_low", "ci_high", "n",
                "ref_sign", "cell"]]


def pivot_cells(table: pd.DataFrame) -> pd.DataFrame:
    """Wide view for the report: rows = claims, columns = variation=level, cells = verdict
    with the estimate."""
    t = table.assign(column=table.variation + "=" + table.level.astype(str),
                     text=[f"{c} ({e:.3f})" if np.isfinite(e) else c
                           for c, e in zip(table.cell, table.estimate, strict=True)])
    order = list(dict.fromkeys(t.column))
    wide = t.pivot(index="claim", columns="column", values="text").reindex(columns=order)
    return wide.fillna("not run")
