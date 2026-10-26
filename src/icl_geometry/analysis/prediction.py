"""Held-out prediction framework (T-033; reused by T-043).

Given a prompt-level table with predictor columns, an outcome, a `split` column (val/test,
fixed by D-009) and a grouping column (query item) for the bootstrap:

* `evaluate`: for each predictor set, fit a standardized logistic (binary outcome) or linear
  (continuous outcome) model on the `fit` split and score it on the `eval` split: AUROC or R^2,
  with a 95% CI by bootstrap over groups.
* `incremental`: the gain of (baseline + predictors) over the baseline alone on the eval
  split, with a paired bootstrap CI over groups.
* `mixed_effects`: statsmodels MixedLM with a random intercept for the query item and a
  variance component for the demo set (crossed effects approximated via vc_formula).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _model(binary: bool):
    est = LogisticRegression(max_iter=2000) if binary else LinearRegression()
    return make_pipeline(StandardScaler(), est)


def _score(binary: bool, y, pred) -> float:
    if binary:
        return float(roc_auc_score(y, pred)) if len(np.unique(y)) > 1 else float("nan")
    return float(r2_score(y, pred))


def _predict(model, X, binary: bool):
    return model.predict_proba(X)[:, 1] if binary else model.predict(X)


def _group_bootstrap(groups: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in uniq}
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq))
        yield np.concatenate([pos[g] for g in draw])


def evaluate(df: pd.DataFrame, predictor_sets: dict[str, list[str]], outcome: str,
             fit_split: str = "val", eval_split: str = "test", group: str = "query_item",
             n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    binary = df[outcome].dropna().isin([0, 1, True, False]).all()
    fit, ev = df[df.split == fit_split], df[df.split == eval_split]
    rows = []
    for name, cols in predictor_sets.items():
        f, e = fit.dropna(subset=cols + [outcome]), ev.dropna(subset=cols + [outcome])
        model = _model(binary).fit(f[cols].to_numpy(), f[outcome].astype(float if not binary
                                                                        else int))
        pred = _predict(model, e[cols].to_numpy(), binary)
        y = e[outcome].astype(int if binary else float).to_numpy()
        point = _score(binary, y, pred)
        boots = [_score(binary, y[i], pred[i])
                 for i in _group_bootstrap(e[group].to_numpy(), n_boot, seed)]
        boots = [b for b in boots if np.isfinite(b)]
        rows.append({"predictors": name, "metric": "auroc" if binary else "r2", "value": point,
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                     "n_fit": len(f), "n_eval": len(e)})
    return pd.DataFrame(rows)


def incremental(df: pd.DataFrame, baseline: list[str], added: list[str], outcome: str,
                fit_split: str = "val", eval_split: str = "test", group: str = "query_item",
                n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    binary = df[outcome].dropna().isin([0, 1, True, False]).all()
    cols = baseline + added
    f = df[df.split == fit_split].dropna(subset=cols + [outcome])
    e = df[df.split == eval_split].dropna(subset=cols + [outcome])
    yf = f[outcome].astype(int if binary else float)
    y = e[outcome].astype(int if binary else float).to_numpy()
    p0 = _predict(_model(binary).fit(f[baseline].to_numpy(), yf), e[baseline].to_numpy(), binary)
    p1 = _predict(_model(binary).fit(f[cols].to_numpy(), yf), e[cols].to_numpy(), binary)
    gain = _score(binary, y, p1) - _score(binary, y, p0)
    boots = []
    for i in _group_bootstrap(e[group].to_numpy(), n_boot, seed):
        boots.append(_score(binary, y[i], p1[i]) - _score(binary, y[i], p0[i]))
    boots = [b for b in boots if np.isfinite(b)]
    return {"gain": gain, "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_le_0": float(np.mean(np.array(boots) <= 0))}


def mixed_effects(df: pd.DataFrame, formula: str, item: str = "query_item",
                  demo_set: str | None = "demo_key"):
    """Linear mixed model, e.g. formula='margin ~ shift_score + m'. Random intercept for the
    query item; demo set as a variance component."""
    import statsmodels.formula.api as smf

    kw = {}
    if demo_set is not None and demo_set in df:
        kw["vc_formula"] = {"demo": f"0 + C({demo_set})"}
    model = smf.mixedlm(formula, df, groups=df[item], **kw)
    return model.fit(method="lbfgs")
