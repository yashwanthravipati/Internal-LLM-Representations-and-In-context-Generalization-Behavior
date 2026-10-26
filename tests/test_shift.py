import numpy as np
import pytest

from icl_geometry.geometry import shift


def make_items(rng, n_per_cell=15, d=60, sep=2.0):
    """Items on a 2x2 design: attribute A on axis 0, attribute B on axis 1, equally strong."""
    a = np.repeat([0, 0, 1, 1], n_per_cell)
    b = np.tile(np.repeat([0, 1], n_per_cell), 2)
    base = rng.standard_normal((len(a), d))
    base[:, 0] += sep * (a - 0.5)
    base[:, 1] += sep * (b - 0.5)
    return base, a, b


def contexts(rng, base, a, n_prompts=4, gain=0.0, noise=0.3):
    """Prompt-level reps: each item appears in n_prompts contexts; the context amplifies the cued
    axis (0) by `gain`."""
    X, ya, items = [], [], []
    for i, x in enumerate(base):
        for _ in range(n_prompts):
            v = x + noise * rng.standard_normal(len(x))
            v[0] += gain * (a[i] - 0.5)
            X.append(v)
            ya.append(a[i])
            items.append(i)
    return np.array(X), np.array(ya), np.array(items)


def test_delta_snr_planted_vs_null():
    rng = np.random.default_rng(0)
    base, a, b = make_items(rng)
    X0, y0, _ = contexts(rng, base, a, gain=0.0)
    X1, y1, _ = contexts(rng, base, a, gain=3.0)
    Xc, yc, _ = contexts(rng, base, a, gain=0.0)
    assert shift.delta_snr(X1, y1, Xc, yc, m=4) > 0.5
    assert abs(shift.delta_snr(X0, y0, Xc, yc, m=4)) < 0.15


def test_selection_index_and_control():
    rng = np.random.default_rng(1)
    base, a, b = make_items(rng, sep=0.0)  # no item structure on B
    base[:, 0] += 6 * (a - 0.5)
    X, ya, items = contexts(rng, base, a)
    yb = b[items]
    out = shift.selection_index(X, ya, yb, groups=items)
    assert out["acc_cued"] > 0.9 and out["acc_uncued"] < 0.7
    assert out["selection_index"] > 0.25
    assert out["acc_control"] < 0.7  # control task can't be memorized across held-out items


def test_rsa_partial():
    rng = np.random.default_rng(2)
    base, a, b = make_items(rng, sep=0.0, d=30)
    base[:, 0] += 4 * (a - 0.5)
    r = shift.rsa_partial(base, a, b)
    assert r["rsa_a"] > 0.3 and abs(r["rsa_b"]) < 0.15


def test_ccgp_abstract_vs_not():
    rng = np.random.default_rng(3)
    base, a, b = make_items(rng, sep=6.0)
    assert shift.ccgp(base, a, b) > 0.9  # factorized code generalizes
    # XOR-like code: the A direction flips with B, so a decoder trained on one B value fails
    xor = rng.standard_normal(base.shape)
    xor[:, 0] += 6 * (a - 0.5) * np.where(b == 1, 1, -1)
    assert shift.ccgp(xor, a, b) < 0.2


def test_shift_scores_sign_and_scale():
    rng = np.random.default_rng(4)
    base, a, b = make_items(rng)
    Xn = base
    Xctx = base.copy()
    Xctx[:, 0] += 1.0 * (a - 0.5) * 2  # move each item 1 unit toward its class
    axis, scale, classes = shift.shift_axis(Xn, a)
    s = shift.shift_scores(Xctx, Xn, a, axis, scale, classes)
    assert (s > 0).mean() > 0.95
    s_null = shift.shift_scores(Xn + 0.0, Xn, a, axis, scale, classes)
    assert np.allclose(s_null, 0)


@pytest.mark.parametrize("seed", [0, 1])
def test_control_labels_constant_within_item(seed):
    groups = np.repeat(np.arange(10), 3)
    y = np.repeat(np.arange(10) % 2, 3)
    c = shift.control_labels(y, groups, seed)
    for g in range(10):
        assert len(set(c[groups == g])) == 1
    assert c.sum() == y.sum()
