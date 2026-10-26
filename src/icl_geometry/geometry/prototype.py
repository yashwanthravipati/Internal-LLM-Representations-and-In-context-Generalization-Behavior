"""Monte Carlo m-shot learners on fixed representations (Sorscher et al. 2022, Methods §L).

Each draw samples m training examples per class without replacement; every remaining example
is a test example (train and test never overlap, like held-out query items in a prompt).
Errors are reported separately for test examples of a and of b.

Rules:
* `prototype`: nearest class mean (Euclidean), the rule the theory describes.
* `exemplar`: nearest single training example (1-NN), an alternative ICL hypothesis.
"""

from __future__ import annotations

import numpy as np


def _draw_splits(n: int, m: int, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """[n_draws, n] permutations; the first m columns are the training indices."""
    if m >= n:
        raise ValueError(f"m={m} leaves no test examples out of {n}")
    return np.argsort(rng.random((n_draws, n)), axis=1)


def _test_mask(perms: np.ndarray, m: int) -> np.ndarray:
    """[n_draws, n] boolean: True where the example is a test example in that draw."""
    mask = np.ones(perms.shape, dtype=bool)
    np.put_along_axis(mask, perms[:, :m], False, axis=1)
    return mask


def prototype_error(
    Xa: np.ndarray,
    Xb: np.ndarray,
    m: int,
    n_draws: int = 1000,
    rng: np.random.Generator | int | None = 0,
    rule: str = "prototype",
) -> tuple[float, float]:
    """Mean m-shot error on held-out examples of a and of b: returns (err_a, err_b).

    Ties (equal distance) count as half an error.
    """
    rng = np.random.default_rng(rng)
    Xa = np.asarray(Xa, dtype=np.float64)
    Xb = np.asarray(Xb, dtype=np.float64)
    pa, pb = _draw_splits(len(Xa), m, n_draws, rng), _draw_splits(len(Xb), m, n_draws, rng)
    ta, tb = _test_mask(pa, m), _test_mask(pb, m)

    if rule == "prototype":
        proto_a = Xa[pa[:, :m]].mean(1)  # [D, N]
        proto_b = Xb[pb[:, :m]].mean(1)
        w = proto_a - proto_b
        c = 0.5 * ((proto_a**2).sum(1) - (proto_b**2).sum(1))
        # h > 0 means "closer to a":  ||x - pb||^2 - ||x - pa||^2 = 2 (x.w - c)
        ha = Xa @ w.T - c  # [Pa, D]
        hb = Xb @ w.T - c
    elif rule == "exemplar":
        X = np.vstack([Xa, Xb])
        sq = (X**2).sum(1)
        d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
        na = len(Xa)
        da = d2[:, pa[:, :m]].min(-1)  # [Pa+Pb, D] distance to the nearest a exemplar
        db = d2[:, na + pb[:, :m]].min(-1)
        h = db - da
        ha, hb = h[:na], h[na:]
    else:
        raise ValueError(f"unknown rule {rule!r}")

    err_a = (ha < 0) + 0.5 * (ha == 0)
    err_b = (hb > 0) + 0.5 * (hb == 0)
    err_a = (err_a.T * ta).sum() / ta.sum()
    err_b = (err_b.T * tb).sum() / tb.sum()
    return float(err_a), float(err_b)
