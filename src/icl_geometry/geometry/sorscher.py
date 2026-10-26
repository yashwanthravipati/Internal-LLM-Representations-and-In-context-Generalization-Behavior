"""Geometric theory of few-shot prototype learning (Sorscher, Ganguli & Sompolinsky 2022, PNAS).

Conventions follow the official code (github.com/bsorsch/geometry-fewshot-learning, commit
fee5d47, function `geometry()` in the macaque / prototype-vs-exemplar notebooks):

* A manifold is a P x N sample matrix X (P examples, N neurons / residual dimensions).
* `R` are the singular values of the centered X, so R_i**2 / P are the covariance eigenvalues
  (paper Eq. 4) and `u_i` (rows of `U`) the covariance eigenvectors.
* Total variance  tr(C_a) = sum_i R_i**2 / P. This is the paper's R_a**2 (mean squared radius
  up to the N factor, which cancels everywhere it matters).
* Signal        ||dx0||**2 = ||x0_a - x0_b||**2 / tr(C_a)                     (Eq. 1)
* Bias          (tr(C_b) / tr(C_a) - 1) / m                                   (Eq. 1)
* Dimension     D_a = (sum R**2)**2 / sum R**4   (participation ratio)        (Eq. 1)
* Signal-noise overlaps (the paper's ||dx0 . U_a||**2 and ||dx0 . U_b||**2):
      csa = sum_i (u_i^a . dx0_hat)**2 R_i^a**2 / sum R^a**2,
      csb = sum_i (u_i^b . dx0_hat)**2 R_i^b**2 / sum R^a**2,
      overlap_a = csa * signal,  overlap_b = csb * signal
* Subspace-subspace overlap (in the official code, absent from the paper's short form of
  Eq. 1):  ss = sum_ij (u_i^a . u_j^b)**2 R_i^a**2 R_j^b**2 / (sum R^a**2)**2

Dominant-terms SNR (what the official code and the paper's figures use):

    SNR_a = 1/2 (signal + bias) / sqrt(1/(D_a m) + overlap_a + overlap_b/m + ss/m)

`full=True` adds the 1/m**2 terms from SI §3C, as in the official k-way code:
    + 1/(2 D_a m**2) + tr(C_b)/tr(C_a) / (2 D_b m**2).

Generalization error: eps_a = H(SNR_a) = norm.sf(SNR_a) (Gaussian tail).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class ManifoldStats:
    center: np.ndarray  # [N]
    R: np.ndarray  # [K] non-zero singular values of the centered samples
    U: np.ndarray  # [K, N] corresponding unit axes (covariance eigenvectors)
    P: int  # number of samples

    @property
    def total_var(self) -> float:
        """tr(C) = sum R_i**2 / P."""
        return float((self.R**2).sum() / self.P)

    @property
    def D(self) -> float:
        """Participation-ratio dimension (sum R**2)**2 / sum R**4. Capped at P - 1. NaN for a
        degenerate manifold (all samples identical)."""
        r2 = self.R**2
        return float(r2.sum() ** 2 / (r2**2).sum()) if r2.size else float("nan")


def manifold_stats(X: np.ndarray, rtol: float = 1e-10) -> ManifoldStats:
    """Centroid, radii and axes of a P x N manifold sample.

    Uses a thin SVD of the centered P x N matrix, which costs O(P**2 N) for P << N (the same as
    the Gram-matrix trick) and never forms the N x N covariance. Keeps only non-zero radii
    (at most P - 1).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"need a P x N matrix with P >= 2, got {X.shape}")
    center = X.mean(0)
    _, R, Vt = np.linalg.svd(X - center, full_matrices=False)
    keep = R > rtol * max(R.max(), 1e-300)
    return ManifoldStats(center=center, R=R[keep], U=Vt[keep], P=X.shape[0])


def pair_terms(a: ManifoldStats, b: ManifoldStats) -> dict[str, float]:
    """m-independent geometric terms for discriminating a (test) from b. All terms are NaN
    when manifold a is degenerate (zero variance), e.g. identical token embeddings."""
    if a.R.size == 0 or b.R.size == 0:
        keys = ("signal", "radius_ratio", "D_a", "D_b", "csa", "csb", "overlap_a",
                "overlap_b", "ss", "sum_r2_ratio")
        return {k: float("nan") for k in keys} | {"P_a": a.P, "P_b": b.P}
    dx0 = a.center - b.center
    dist2 = float(dx0 @ dx0)
    sum_r2_a = float((a.R**2).sum())
    sum_r2_b = float((b.R**2).sum())
    signal = dist2 / a.total_var
    dx0_hat = dx0 / np.sqrt(dist2) if dist2 > 0 else dx0
    csa = float(((a.U @ dx0_hat) ** 2 * a.R**2).sum() / sum_r2_a)
    csb = float(((b.U @ dx0_hat) ** 2 * b.R**2).sum() / sum_r2_a)
    cosphi = a.U @ b.U.T
    ss = float((cosphi**2 * a.R[:, None] ** 2 * b.R[None, :] ** 2).sum() / sum_r2_a**2)
    return {
        "signal": signal,
        "radius_ratio": b.total_var / a.total_var,
        "D_a": a.D,
        "D_b": b.D,
        "csa": csa,
        "csb": csb,
        "overlap_a": csa * signal,
        "overlap_b": csb * signal,
        "ss": ss,
        "P_a": a.P,
        "P_b": b.P,
        # sum R**2 ratio as the official code computes the bias (P-weighted totals). Equal to
        # radius_ratio when P_a == P_b.
        "sum_r2_ratio": sum_r2_b / sum_r2_a,
    }


def snr_from_terms(
    t: dict[str, float], m: float | np.ndarray, full: bool = False
) -> dict[str, np.ndarray]:
    """SNR_a (Eq. 1) and its decomposition for one or many m. `m=np.inf` gives the limit."""
    m = np.asarray(m, dtype=np.float64)
    inv_m = np.where(np.isinf(m), 0.0, 1.0 / m)
    bias = (t["radius_ratio"] - 1.0) * inv_m
    dim_term = inv_m / t["D_a"]
    noise2 = dim_term + t["overlap_a"] + t["overlap_b"] * inv_m + t["ss"] * inv_m
    if full:
        noise2 = noise2 + inv_m**2 / (2 * t["D_a"]) + t["radius_ratio"] * inv_m**2 / (2 * t["D_b"])
    snr = 0.5 * (t["signal"] + bias) / np.sqrt(noise2)
    return {
        "snr": snr,
        "error": norm.sf(snr),
        "signal": np.broadcast_to(t["signal"], m.shape).copy(),
        "bias": bias,
        "dim_term": dim_term,
        "overlap_a_term": np.broadcast_to(t["overlap_a"], m.shape).copy(),
        "overlap_b_term": t["overlap_b"] * inv_m,
        "ss_term": t["ss"] * inv_m,
    }


def snr(
    a: ManifoldStats, b: ManifoldStats, m: float | np.ndarray, full: bool = False
) -> dict[str, np.ndarray]:
    """SNR and predicted error for classifying test examples of `a` against `b` (Eq. 1)."""
    return snr_from_terms(pair_terms(a, b), m, full=full)


def pairwise_snr(
    manifolds: list[np.ndarray] | list[ManifoldStats], m: float, full: bool = False
) -> np.ndarray:
    """K x K matrix with SNR[a, b] = SNR of test examples of a vs b; NaN on the diagonal."""
    stats = [s if isinstance(s, ManifoldStats) else manifold_stats(s) for s in manifolds]
    k = len(stats)
    out = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i != j:
                out[i, j] = snr(stats[i], stats[j], m, full=full)["snr"]
    return out


def random_projection_snr(snr_inf: float | np.ndarray, D_inf: float, M: float | np.ndarray):
    """Eq. 2: expected SNR after projecting onto M random dimensions,
    SNR(M) = SNR_inf / (1 + D_inf / M)  (equivalently D(M)**-1 = D_inf**-1 + M**-1)."""
    return np.asarray(snr_inf) / (1.0 + D_inf / np.asarray(M, dtype=np.float64))


def projected_dimension(D_inf: float, M: float | np.ndarray):
    """Eq. 2 companion: D(M) = 1 / (1/D_inf + 1/M)."""
    return 1.0 / (1.0 / D_inf + 1.0 / np.asarray(M, dtype=np.float64))
