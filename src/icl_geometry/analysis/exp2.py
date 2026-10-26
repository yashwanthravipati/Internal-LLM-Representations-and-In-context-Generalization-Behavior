"""Experiment 2 analysis helpers: layer-wise shift metrics (H2, T-032) and per-prompt shift
scores (H3, T-033).

Data model: `meta` is a prompt table (one row per cached activation row) with columns
prompt_id, domain, cued_attr, uncued_attr, m, condition, query_item, query_attrs (dict),
congruent, plus behavior columns. Activations are [n_prompts, n_states, d] arrays.

Every in-context quantity is compared with the **random-label control**: the same demo
items, labels permuted so that no attribute predicts them (PLAN §6.4). The context itself
(more tokens, label tokens) changes residual streams; only the cued *structure* differs
between main and control.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.geometry import shift
from icl_geometry.geometry.shift import shift_axis, shift_scores


def attr_values(meta: pd.DataFrame, attr_col: str) -> np.ndarray:
    """Query value of the attribute named in column `attr_col` (cued_attr or uncued_attr)."""
    return np.array([q[a] for q, a in zip(meta.query_attrs, meta[attr_col], strict=True)])


def cell_metrics(X_main: np.ndarray, meta_main: pd.DataFrame, X_ctrl: np.ndarray,
                 meta_ctrl: pd.DataFrame, m: float) -> dict[str, float]:
    """Metrics for one (domain, cued attribute, m, position, layer) cell."""
    yc_main, yu_main = attr_values(meta_main, "cued_attr"), attr_values(meta_main, "uncued_attr")
    yc_ctrl, yu_ctrl = attr_values(meta_ctrl, "cued_attr"), attr_values(meta_ctrl, "uncued_attr")
    g_main, g_ctrl = meta_main.query_item.to_numpy(), meta_ctrl.query_item.to_numpy()
    out: dict[str, float] = {}
    out["snr_cued_main"] = shift.snr_partition(X_main, yc_main, m)
    out["snr_cued_ctrl"] = shift.snr_partition(X_ctrl, yc_ctrl, m)
    out["snr_uncued_main"] = shift.snr_partition(X_main, yu_main, m)
    out["snr_uncued_ctrl"] = shift.snr_partition(X_ctrl, yu_ctrl, m)
    out["delta_snr_cued"] = out["snr_cued_main"] - out["snr_cued_ctrl"]
    out["delta_snr_uncued"] = out["snr_uncued_main"] - out["snr_uncued_ctrl"]
    for tag, X, yc, yu, g in (("main", X_main, yc_main, yu_main, g_main),
                              ("ctrl", X_ctrl, yc_ctrl, yu_ctrl, g_ctrl)):
        si = shift.selection_index(X, yc, yu, groups=g)
        out[f"acc_cued_{tag}"] = si["acc_cued"]
        out[f"acc_uncued_{tag}"] = si["acc_uncued"]
        out[f"acc_control_task_{tag}"] = si.get("acc_control", np.nan)
        out[f"ccgp_cued_{tag}"] = shift.ccgp(X, yc, yu)
    out["delta_acc_cued"] = out["acc_cued_main"] - out["acc_cued_ctrl"]
    out["delta_acc_uncued"] = out["acc_uncued_main"] - out["acc_uncued_ctrl"]
    out["n_main"], out["n_ctrl"] = len(meta_main), len(meta_ctrl)
    return out


def layer_profile(acts: np.ndarray, meta: pd.DataFrame, layers: list[int],
                  group_cols=("domain", "cued_attr", "m")) -> pd.DataFrame:
    """Run `cell_metrics` for every group and layer (main vs random_label rows of `meta`,
    whose index aligns with `acts` rows)."""
    rows = []
    for key, g in meta.groupby(list(group_cols)):
        gm = g[g.condition == "main"]
        gc = g[g.condition == "random_label"]
        if len(gm) < 8 or len(gc) < 8:
            continue
        m = float(dict(zip(group_cols, key, strict=True)).get("m", 4))
        for layer in layers:
            Xm = acts[gm.index.to_numpy(), layer].astype(np.float64)
            Xc = acts[gc.index.to_numpy(), layer].astype(np.float64)
            rec = dict(zip(group_cols, key, strict=True))
            rec["layer"] = layer
            rec |= cell_metrics(Xm, gm, Xc, gc, m)
            rows.append(rec)
    return pd.DataFrame(rows)


def load_run(run_dir: str):
    run = io.REPO_ROOT / run_dir
    cfg = io.RunConfig.from_yaml(run / "config.yaml")
    meta = json.loads((run / "metadata.json").read_text())
    prompts = pd.DataFrame(io.read_jsonl(io.REPO_ROOT / cfg.prompt_set))
    beh = pd.read_parquet(run / "behavior.parquet")
    cache = io.REPO_ROOT / meta["activation_cache"]
    acts = {p: io.load_activations(cache, p)[0] for p in ("query_item", "decision")}
    index = io.load_activations(cache, "query_item")[1]
    df = index.merge(prompts, on="prompt_id").merge(beh, on="prompt_id").sort_values("row")
    df = df.reset_index(drop=True)
    df["cued_value"] = attr_values(df, "cued_attr")
    return cfg, df, acts


def neutral_lookup(model: str):
    base = io.CACHE_DIR / io.model_short_name(model)
    d = sorted(p for p in base.glob("neutral_*") if not p.name.endswith(".partial"))[-1]
    states = np.load(d / "states.npy", mmap_mode="r")
    index = pd.read_parquet(d / "index.parquet")
    return states, dict(zip(index.item, index.row, strict=True))


def neutral_matrix(df: pd.DataFrame, neutral, row_of, layer: int) -> np.ndarray:
    """Neutral rep (mean over carriers) of each row's query item at `layer`."""
    return np.stack([neutral[row_of[i], :, layer].astype(np.float64).mean(0)
                     for i in df.query_item])


def shift_feature(df: pd.DataFrame, acts: np.ndarray, neutral, row_of, layer: int,
                  fit_mask: np.ndarray) -> np.ndarray:
    """Shift score per row of df at `layer`, axes fit on rows where fit_mask is True."""
    return shift_feature_from(df, acts[:, layer].astype(np.float64),
                              neutral_matrix(df, neutral, row_of, layer), fit_mask)


def shift_feature_from(df: pd.DataFrame, X: np.ndarray, Xn: np.ndarray,
                       fit_mask: np.ndarray) -> np.ndarray:
    """Shift scores from in-context reps X and neutral reps Xn (any representation space):
    per (domain, cued attribute) axis fit on main-condition rows where fit_mask is True."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby(["domain", "cued_attr"]):
        idx = g.index.to_numpy()
        fit = idx[fit_mask[idx] & (df.condition.to_numpy()[idx] == "main")]
        if len(np.unique(df.cued_value.to_numpy()[fit])) < 2:
            continue
        axis, scale, classes = shift_axis(X[fit], df.cued_value.to_numpy()[fit])
        out[idx] = shift_scores(X[idx], Xn[idx], df.cued_value.to_numpy()[idx], axis, scale,
                                classes)
    return out
