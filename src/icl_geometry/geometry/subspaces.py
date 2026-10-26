"""Dimension-matched comparison subspaces for the J-space test (H5, T-043).

Any "J-space predicts better" claim must beat subspaces of the same dimension M:
* random orthonormal subspaces (expected SNR loss given by Sorscher Eq. 2),
* the top-M principal components (fit on training data only),
* supervised directions (upper reference; fit on training data only).
"""

from __future__ import annotations

import numpy as np


def random_basis(d: int, M: int, rng: np.random.Generator | int | None = 0) -> np.ndarray:
    """[d, M] orthonormal basis of a uniformly random M-dim subspace."""
    rng = np.random.default_rng(rng)
    q, _ = np.linalg.qr(rng.standard_normal((d, M)))
    return q


def pca_basis(X_train: np.ndarray, M: int) -> np.ndarray:
    """[d, M] top principal axes of X_train (centered)."""
    Xc = X_train - X_train.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return vt[:M].T


def mean_difference_basis(X_train: np.ndarray, labels: list[np.ndarray]) -> np.ndarray:
    """Orthonormalized class-mean-difference directions for one or more binary labelings
    (supervised upper reference)."""
    dirs = []
    for y in labels:
        classes = np.unique(y)
        dirs.append(X_train[y == classes[1]].mean(0) - X_train[y == classes[0]].mean(0))
    q, _ = np.linalg.qr(np.stack(dirs, 1))
    return q


def project(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return X @ basis
