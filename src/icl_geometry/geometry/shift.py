"""Exp 2 shift metrics (T-031): does the in-context representation of the same items reorganize
toward the demonstrated (cued) grouping?

All functions take plain arrays: X [n, d] (one representation per prompt or per item), and
binary attribute labels. Prompt-level data must be split by *item* in cross-validation
(`groups`), because the same item appears in many prompts.

* `snr_partition` / `delta_snr`: Sorscher SNR of the cued partition (Eq. 1, mean of both
  directions), in context minus control (neutral or random-label reps).
* `probe_accuracy` / `selection_index`: cross-validated decoding accuracy (nearest-class-mean
  by default; regularized logistic optional) for the cued minus the uncued attribute, with a
  Hewitt & Liang (2019) control task (labels randomly reassigned per item) to calibrate
  selectivity.
* `rsa_partial`: partial Spearman correlation of the item RDM with each attribute's model RDM,
  controlling for the other.
* `ccgp`: cross-condition generalization (Bernardi et al. 2020): train the cued decoder on one
  value of the uncued attribute, test on the other.
* `shift_axis` / `shift_scores`: per-prompt projection of (in-context - neutral) onto the cued
  axis (difference of class means, fit on training prompts only), signed so that positive
  means "moved toward the correct class", in units of half the class-mean separation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icl_geometry.geometry.sorscher import manifold_stats, snr


def snr_partition(X: np.ndarray, y: np.ndarray, m: float, full: bool = True) -> float:
    """Mean of SNR(a vs b) and SNR(b vs a) for the binary partition y of X."""
    y = np.asarray(y)
    a, b = np.unique(y)
    sa, sb = manifold_stats(X[y == a]), manifold_stats(X[y == b])
    return float(0.5 * (snr(sa, sb, m, full=full)["snr"] + snr(sb, sa, m, full=full)["snr"]))


def delta_snr(X_ctx, y_ctx, X_ctrl, y_ctrl, m: float, full: bool = True) -> float:
    return snr_partition(X_ctx, y_ctx, m, full) - snr_partition(X_ctrl, y_ctrl, m, full)


DEFAULT_C = 0.05  # strong L2 regularization: few samples (tens of items) in d >= 1024 dims


def _probe(C: float = DEFAULT_C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def _fit_predict(Xtr, ytr, Xte, decoder: str, C: float):
    if decoder == "prototype":
        classes = np.unique(ytr)
        mu = np.stack([Xtr[ytr == c].mean(0) for c in classes])
        d2 = (Xte**2).sum(1)[:, None] - 2 * Xte @ mu.T + (mu**2).sum(1)[None]
        return classes[d2.argmin(1)]
    return _probe(C).fit(Xtr, ytr).predict(Xte)


def probe_accuracy(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
                   n_splits: int = 5, decoder: str = "prototype", C: float = DEFAULT_C,
                   seed: int = 0) -> float:
    """Cross-validated decoding accuracy; folds split by `groups` (e.g. item) when given.

    `prototype` (nearest class mean) is the default: with tens of items in >= 1024 dimensions a
    logistic probe overfits item identity and generalizes worse to held-out items (see
    tests/test_shift.py). `logistic` is the L2-regularized alternative."""
    y = np.asarray(y)
    if groups is not None:
        splits = GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    correct, total = 0, 0
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        correct += (_fit_predict(X[tr], y[tr], X[te], decoder, C) == y[te]).sum()
        total += len(te)
    return float(correct / total) if total else float("nan")


def control_labels(y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Hewitt & Liang control task: each group (item) gets a random label, with the same
    class balance as y, constant across that item's prompts."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    item_label = {g: y[groups == g][0] for g in uniq}
    shuffled = rng.permutation([item_label[g] for g in uniq])
    lookup = dict(zip(uniq, shuffled, strict=True))
    return np.array([lookup[g] for g in groups])


def selection_index(X, y_cued, y_uncued, groups=None, decoder: str = "prototype",
                    C: float = DEFAULT_C) -> dict[str, float]:
    acc_c = probe_accuracy(X, y_cued, groups, decoder=decoder, C=C)
    acc_u = probe_accuracy(X, y_uncued, groups, decoder=decoder, C=C)
    out = {"acc_cued": acc_c, "acc_uncued": acc_u, "selection_index": acc_c - acc_u}
    if groups is not None:
        acc_ctrl = probe_accuracy(X, control_labels(np.asarray(y_cued), np.asarray(groups)),
                                  groups, decoder=decoder, C=C)
        out["acc_control"] = acc_ctrl
        out["selectivity_cued"] = acc_c - acc_ctrl
    return out


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    r = np.corrcoef([rx, ry, rz])
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def rsa_partial(X_items: np.ndarray, attr_a: np.ndarray, attr_b: np.ndarray) -> dict[str, float]:
    """X_items [n_items, d] (e.g. reps averaged per item within a context)."""
    rdm = pdist(X_items, "correlation")
    ma = pdist(np.asarray(attr_a)[:, None], lambda u, v: float(u[0] != v[0]))
    mb = pdist(np.asarray(attr_b)[:, None], lambda u, v: float(u[0] != v[0]))
    return {"rsa_a": _partial_spearman(rdm, ma, mb), "rsa_b": _partial_spearman(rdm, mb, ma)}


def ccgp(X: np.ndarray, y_cued: np.ndarray, y_uncued: np.ndarray,
         decoder: str = "prototype", C: float = DEFAULT_C) -> float:
    """Train on one uncued value, test on the other (both ways, averaged). The default
    `prototype` decoder (nearest class mean) is robust with few samples in high dimensions;
    `logistic` uses the regularized probe."""
    y_cued, y_uncued = np.asarray(y_cued), np.asarray(y_uncued)
    accs = []
    for v in np.unique(y_uncued):
        tr, te = y_uncued == v, y_uncued != v
        if len(np.unique(y_cued[tr])) < 2:
            continue
        pred = _fit_predict(X[tr], y_cued[tr], X[te], decoder, C)
        accs.append((pred == y_cued[te]).mean())
    return float(np.mean(accs)) if accs else float("nan")


def shift_axis(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, float, object]:
    """Unit cued axis (mean of class 1 - mean of class 0), half-separation scale, classes."""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    diff = X_train[y_train == classes[1]].mean(0) - X_train[y_train == classes[0]].mean(0)
    norm = float(np.linalg.norm(diff))
    return diff / norm, norm / 2, classes


def shift_scores(X_ctx: np.ndarray, X_neutral: np.ndarray, y: np.ndarray, axis: np.ndarray,
                 scale: float, classes) -> np.ndarray:
    """Signed projection of (X_ctx - X_neutral) on the axis, positive = toward the item's own
    cued class, in units of half the class-mean separation."""
    sign = np.where(np.asarray(y) == classes[1], 1.0, -1.0)
    return sign * ((X_ctx - X_neutral) @ axis) / scale
