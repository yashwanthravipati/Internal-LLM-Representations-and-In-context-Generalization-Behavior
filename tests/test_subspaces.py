import numpy as np

from icl_geometry.geometry.sorscher import manifold_stats, random_projection_snr, snr
from icl_geometry.geometry.subspaces import pca_basis, project, random_basis


def test_random_projection_matches_eq2():
    """SNR on M random dimensions ~ SNR_inf / (1 + D_inf / M) (Sorscher Eq. 2) in the m -> inf
    limit with M << d, averaged over random subspaces. (At finite m the measured loss is
    smaller than Eq. 2 predicts; T-043 uses empirical random subspaces as the null.)"""
    rng = np.random.default_rng(0)
    d, P = 400, 3000
    radii = np.ones(100)
    axes_a = np.linalg.qr(rng.standard_normal((d, 100)))[0]
    axes_b = np.linalg.qr(rng.standard_normal((d, 100)))[0]
    shift = np.zeros(d)
    shift[:50] = 0.25
    Xa = (rng.standard_normal((P, 100)) * radii) @ axes_a.T
    Xb = shift + (rng.standard_normal((P, 100)) * radii) @ axes_b.T
    sa, sb = manifold_stats(Xa), manifold_stats(Xb)
    s_inf = float(snr(sa, sb, np.inf)["snr"])
    for M in (40, 80, 160):
        vals = []
        for seed in range(8):
            B = random_basis(d, M, seed)
            pa, pb = manifold_stats(project(Xa, B)), manifold_stats(project(Xb, B))
            vals.append(float(snr(pa, pb, np.inf)["snr"]))
        pred = random_projection_snr(s_inf, sa.D, M)
        assert abs(np.mean(vals) - pred) / pred < 0.2, (M, np.mean(vals), pred)


def test_pca_basis_orthonormal():
    X = np.random.default_rng(1).standard_normal((50, 30))
    B = pca_basis(X, 5)
    np.testing.assert_allclose(B.T @ B, np.eye(5), atol=1e-10)
