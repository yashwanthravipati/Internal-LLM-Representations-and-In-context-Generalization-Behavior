"""Experiment 1: does Sorscher-theory error from neutral representations predict LLM few-shot
error across tasks (category pair x direction x m)? (T-024; PLAN §6.9 item 1.)

A *task* is (a, b, m): test examples of category a against category b with m demos per
category. LLM error = mean error on prompts whose query is from a. Theory error = H(SNR_a)
from the neutral-representation manifolds of a and b at a given layer. Prototype error = Monte
Carlo nearest-prototype learner on the same manifolds.

Statistics: Spearman rho across tasks per layer; the layer is selected on the val split and
reported on test; CIs by bootstrap over *categories* (tasks share categories); the null is a
permutation of LLM errors across tasks.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from icl_geometry.geometry.prototype import prototype_error
from icl_geometry.geometry.sorscher import manifold_stats, pair_terms, snr_from_terms


def llm_task_errors(prompts: pd.DataFrame, behavior: pd.DataFrame) -> pd.DataFrame:
    """One row per (a, b, m): error rates (raw and calibrated), mean margin, n prompts."""
    df = behavior.merge(prompts, on="prompt_id")
    df["a"] = df.query_attrs.map(lambda q: q["category"])
    df["b"] = [p[1] if p[0] == a else p[0] for p, a in zip(df.category_pair, df.a, strict=True)]
    agg = {"error": ("correct", lambda s: 1 - s.mean()), "margin": ("margin", "mean"),
           "label_mass": ("label_mass", "mean"), "n": ("correct", "size")}
    if "correct_calibrated" in df:
        agg["error_cal"] = ("correct_calibrated", lambda s: 1 - s.mean())
    out = df.groupby(["a", "b", "m"]).agg(**agg).reset_index()
    out["related"] = df.groupby(["a", "b", "m"]).related.first().values
    return out


def theory_table(reps: dict[str, np.ndarray], tasks: pd.DataFrame, full: bool = True,
                 n_draws: int = 0, seed: int = 0) -> pd.DataFrame:
    """Theory (and optionally Monte Carlo prototype) error for each task row (a, b, m) given
    per-category representation matrices reps[c] = [P_c, d] at one layer."""
    stats = {c: manifold_stats(X) for c, X in reps.items()}
    rows = []
    cache: dict[tuple[str, str], dict] = {}
    for t in tasks.itertuples(index=False):
        key = (t.a, t.b)
        if key not in cache:
            cache[key] = pair_terms(stats[t.a], stats[t.b])
        terms = cache[key]
        out = snr_from_terms(terms, t.m, full=full)
        row = {"a": t.a, "b": t.b, "m": t.m, "snr": float(out["snr"]),
               "theory_error": float(out["error"]), "signal": terms["signal"],
               "bias": float(out["bias"]), "D_a": terms["D_a"],
               "overlap_a": terms["overlap_a"], "P_a": terms["P_a"],
               "centroid_dist2": terms["signal"] * stats[t.a].total_var}
        if n_draws:
            ea, _ = prototype_error(reps[t.a], reps[t.b], int(t.m), n_draws=n_draws,
                                    rng=seed + zlib.crc32(f"{t.a}|{t.b}|{t.m}".encode()))
            row["proto_error"] = ea
        rows.append(row)
    return pd.DataFrame(rows)


def spearman(x, y) -> float:
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def bootstrap_over_categories(tasks: pd.DataFrame, xcol: str, ycol: str, n_boot: int = 1000,
                              seed: int = 0) -> tuple[float, float]:
    """95% CI of Spearman rho, resampling categories with replacement; a task (a, b) gets weight
    count(a) * count(b) in the resample (implemented by row repetition)."""
    rng = np.random.default_rng(seed)
    cats = np.array(sorted(set(tasks.a) | set(tasks.b)))
    vals = []
    for _ in range(n_boot):
        counts = pd.Series(rng.choice(cats, len(cats))).value_counts()
        w = tasks.a.map(counts).fillna(0) * tasks.b.map(counts).fillna(0)
        rep = tasks.loc[tasks.index.repeat(w.astype(int))]
        if rep[xcol].nunique() > 1 and rep[ycol].nunique() > 1:
            vals.append(spearman(rep[xcol], rep[ycol]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def permutation_null(x, y, n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return np.array([spearman(x, rng.permutation(y)) for _ in range(n_perm)])
