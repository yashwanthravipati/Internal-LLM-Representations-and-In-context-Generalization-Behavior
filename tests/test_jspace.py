import itertools

import numpy as np
import torch
from scipy.optimize import nnls

from icl_geometry.lens.jspace import gradient_pursuit, span_projection, variance_explained


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def test_recovers_planted_sparse_code():
    g = torch.Generator().manual_seed(0)
    d, T, k = 256, 2000, 5
    atoms = unit(torch.randn(T, d, generator=g))
    supports = [torch.randperm(T, generator=g)[:k] for _ in range(20)]
    coefs = [torch.rand(k, generator=g) + 0.5 for _ in range(20)]
    h = torch.stack([(c[:, None] * atoms[s]).sum(0) for s, c in zip(supports, coefs,
                                                                     strict=True)])
    code = gradient_pursuit(h, atoms, k=k)
    for i, s in enumerate(supports):
        assert set(code.indices[i].tolist()) == set(s.tolist())
    assert variance_explained(h, code.recon) > 0.99
    assert (code.coefs >= 0).all()


def test_matches_brute_force_small():
    """Tiny dictionary: compare the greedy support with the best non-negative 2-subset (exhaustive
    NNLS). Greedy pursuit isn't guaranteed optimal, so require near-optimal in most cases."""
    rng = np.random.default_rng(1)
    d, T = 6, 8
    A = rng.standard_normal((T, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    near_opt = 0
    for _ in range(50):
        h = rng.standard_normal(d)
        best = min(nnls(A[list(s)].T, h)[1] for s in itertools.combinations(range(T), 2))
        code = gradient_pursuit(torch.tensor(h[None]).float(), torch.tensor(A).float(), k=2)
        sup = [i for i in code.indices[0].tolist() if i >= 0]
        err_gp = nnls(A[sup].T, h)[1] if sup else np.linalg.norm(h)
        assert err_gp >= best - 1e-6
        near_opt += err_gp <= best * 1.1 + 1e-6
    assert near_opt >= 35  # >= 70% of cases within 10% of the optimum


def test_span_projection_rank():
    v = torch.randn(3, 20)
    v = torch.cat([v, v[:1] * 2])  # rank 3
    assert span_projection(torch.randn(5, 20), v).shape == (5, 3)
