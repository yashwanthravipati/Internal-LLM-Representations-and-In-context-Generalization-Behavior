import numpy as np
import pytest

from icl_geometry.geometry.prototype import prototype_error
from icl_geometry.geometry.sorscher import (
    manifold_stats,
    pairwise_snr,
    projected_dimension,
    random_projection_snr,
    snr,
)


def official_geometry(centers, Rs, Us, m):
    """Verbatim numpy port of `geometry()` from bsorsch/geometry-fewshot-learning
    (Prototype vs exemplar theory.ipynb, commit fee5d47)."""
    K = len(centers)
    P = Rs.shape[1]
    dists = np.sqrt(((centers[:, None] - centers[None]) ** 2).sum(-1))
    dist_norm = dists / np.sqrt((Rs**2).sum(-1)[:, None] / P)
    Ds = np.sum(Rs**2, axis=-1) ** 2 / np.sum(Rs**4, axis=-1)
    csa, csb, ss = np.full((K, K), np.nan), np.full((K, K), np.nan), np.full((K, K), np.nan)
    for a in range(K):
        for b in range(K):
            if a != b:
                dx0 = centers[a] - centers[b]
                dx0hat = dx0 / np.linalg.norm(dx0)
                csa[a, b] = ((Us[a] @ dx0hat) ** 2 * Rs[a] ** 2).sum() / (Rs[a] ** 2).sum()
                csb[a, b] = ((Us[b] @ dx0hat) ** 2 * Rs[b] ** 2).sum() / (Rs[a] ** 2).sum()
                cosphi = Us[a] @ Us[b].T
                ss[a, b] = (cosphi**2 * Rs[a][:, None] ** 2 * Rs[b] ** 2).sum() / (
                    Rs[a] ** 2
                ).sum() ** 2
    css = (csa + csb / m) * dist_norm**2
    bias = (Rs**2).sum(-1) / (Rs**2).sum(-1)[:, None] - 1
    return 1 / 2 * (dist_norm**2 + bias / m) / np.sqrt(1 / Ds[:, None] / m + css + ss / m)


def gaussian_manifold(rng, n_dim, center, radii, axes=None, n_samples=2000):
    z = rng.standard_normal((n_samples, len(radii))) * radii
    if axes is None:
        axes = np.linalg.qr(rng.standard_normal((n_dim, len(radii))))[0]
    return center + z @ axes.T


def test_manifold_stats_basic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 100))
    s = manifold_stats(X)
    assert s.P == 30 and len(s.R) == 29  # at most P - 1 non-zero radii
    np.testing.assert_allclose(s.total_var, np.trace(np.cov(X.T, bias=True)), rtol=1e-10)
    assert 1 <= s.D <= 29
    # orthonormal axes
    np.testing.assert_allclose(s.U @ s.U.T, np.eye(29), atol=1e-10)


def test_participation_ratio_known():
    # Equal variance along k orthogonal axes -> D = k (large-sample limit).
    rng = np.random.default_rng(1)
    X = gaussian_manifold(rng, 50, np.zeros(50), np.ones(5), n_samples=20000)
    assert manifold_stats(X).D == pytest.approx(5, rel=0.02)


def test_matches_official_code():
    rng = np.random.default_rng(2)
    P, N, K, m = 40, 60, 4, 3
    manifolds = [rng.standard_normal((P, N)) * rng.uniform(0.5, 2, N) + rng.standard_normal(N)
                 for _ in range(K)]
    centers, Rs, Us = [], [], []
    for X in manifolds:
        c = X.mean(0)
        _, R, Vt = np.linalg.svd(X - c, full_matrices=False)
        centers.append(c)
        Rs.append(R)
        Us.append(Vt)
    ref = official_geometry(np.stack(centers), np.stack(Rs), np.stack(Us), m)
    ours = pairwise_snr(manifolds, m)
    mask = ~np.eye(K, dtype=bool)
    np.testing.assert_allclose(ours[mask], ref[mask], rtol=1e-8)


@pytest.mark.parametrize(
    "name",
    ["isotropic", "unequal_radii", "signal_noise_overlap"],
)
def test_theory_matches_monte_carlo_high_d(name):
    """Theory vs simulated prototype learning in the regime the paper claims accuracy for
    (high D, many samples). Tolerance 0.03 absolute error covers Monte Carlo noise (the
    per-draw prototype noise is shared by all test items, so draws dominate the variance)."""
    rng = np.random.default_rng(3)
    n = 150
    e0 = np.eye(n)[0]
    if name == "isotropic":
        Xa = gaussian_manifold(rng, n, 0, np.ones(80))
        Xb = gaussian_manifold(rng, n, 2.2 * e0, np.ones(80))
    elif name == "unequal_radii":
        Xa = gaussian_manifold(rng, n, 0, np.ones(80))
        Xb = gaussian_manifold(rng, n, 3.0 * e0, 1.4 * np.ones(80))
    else:
        axes = np.eye(n)[:, :40]
        radii = np.r_[2.5, np.ones(39)]
        Xa = gaussian_manifold(rng, n, 0, radii, axes)
        Xb = gaussian_manifold(rng, n, 2.0 * e0, radii, axes)
    sa, sb = manifold_stats(Xa), manifold_stats(Xb)
    for m in (1, 2, 5):
        mc_a, mc_b = prototype_error(Xa, Xb, m, n_draws=1500, rng=4)
        th_a = float(snr(sa, sb, m, full=True)["error"])
        th_b = float(snr(sb, sa, m, full=True)["error"])
        assert abs(mc_a - th_a) < 0.03, (name, m, mc_a, th_a)
        assert abs(mc_b - th_b) < 0.03, (name, m, mc_b, th_b)


def test_m_infinity_limit_and_decomposition():
    rng = np.random.default_rng(5)
    a, b = manifold_stats(rng.standard_normal((50, 80))), manifold_stats(
        rng.standard_normal((50, 80)) + 0.5
    )
    out = snr(a, b, np.array([1.0, 10.0, 1e6, np.inf]))
    # bias, dimension, overlap_b and ss terms vanish as m -> inf
    assert out["bias"][-1] == 0 and out["dim_term"][-1] == 0 and out["ss_term"][-1] == 0
    np.testing.assert_allclose(out["snr"][2], out["snr"][3], rtol=1e-4)
    lim = 0.5 * out["signal"][-1] / np.sqrt(out["overlap_a_term"][-1])
    np.testing.assert_allclose(out["snr"][-1], lim)


def test_bias_predicts_asymmetry():
    # The larger manifold is misclassified more often (bias term), as in the paper.
    rng = np.random.default_rng(6)
    small = manifold_stats(gaussian_manifold(rng, 100, 0, np.ones(60), n_samples=500))
    big = manifold_stats(gaussian_manifold(rng, 100, 2 * np.eye(100)[0], 1.5 * np.ones(60),
                                           n_samples=500))
    assert snr(big, small, 2)["error"] > snr(small, big, 2)["error"]


def test_random_projection_eq2():
    np.testing.assert_allclose(random_projection_snr(2.0, 10.0, 10.0), 1.0)
    np.testing.assert_allclose(projected_dimension(10.0, np.inf), 10.0)
    np.testing.assert_allclose(projected_dimension(10.0, 10.0), 5.0)


def test_degenerate_manifold_gives_nan():
    a = manifold_stats(np.ones((5, 3)))
    b = manifold_stats(np.random.default_rng(0).standard_normal((5, 3)))
    assert np.isnan(snr(a, b, 2)["snr"]) and np.isnan(a.D)
