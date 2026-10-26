import numpy as np
import pytest

from icl_geometry.geometry.prototype import prototype_error


def test_separable_and_chance():
    rng = np.random.default_rng(0)
    Xa = rng.standard_normal((40, 20))
    assert prototype_error(Xa, Xa + 100.0, 2, n_draws=50) == (0.0, 0.0)
    same_a, same_b = prototype_error(Xa, rng.standard_normal((40, 20)), 1, n_draws=2000)
    assert same_a == pytest.approx(0.5, abs=0.05) and same_b == pytest.approx(0.5, abs=0.05)


def test_deterministic_with_seed():
    rng = np.random.default_rng(1)
    Xa, Xb = rng.standard_normal((30, 10)), rng.standard_normal((30, 10)) + 0.5
    assert prototype_error(Xa, Xb, 3, rng=7) == prototype_error(Xa, Xb, 3, rng=7)


def test_matches_naive_loop():
    rng = np.random.default_rng(2)
    Xa, Xb = rng.standard_normal((12, 5)), rng.standard_normal((15, 5)) + 0.7
    m, n_draws = 3, 200
    for rule in ("prototype", "exemplar"):
        fast = prototype_error(Xa, Xb, m, n_draws=n_draws, rng=11, rule=rule)
        # Recreate the same draws and classify one by one.
        r = np.random.default_rng(11)
        pa = np.argsort(r.random((n_draws, len(Xa))), axis=1)
        pb = np.argsort(r.random((n_draws, len(Xb))), axis=1)
        errs_a, errs_b = [], []
        for d in range(n_draws):
            tra, tea = Xa[pa[d, :m]], Xa[pa[d, m:]]
            trb, teb = Xb[pb[d, :m]], Xb[pb[d, m:]]
            if rule == "prototype":
                def dist(x, tr):
                    return ((x[:, None] - tr.mean(0)[None]) ** 2).sum(-1)[:, 0]
            else:
                def dist(x, tr):
                    return ((x[:, None] - tr[None]) ** 2).sum(-1).min(1)
            errs_a += list(dist(tea, tra) > dist(tea, trb))
            errs_b += list(dist(teb, trb) > dist(teb, tra))
        assert fast[0] == pytest.approx(np.mean(errs_a))
        assert fast[1] == pytest.approx(np.mean(errs_b))


def test_m_too_large():
    with pytest.raises(ValueError):
        prototype_error(np.zeros((3, 2)), np.ones((3, 2)), 3)
