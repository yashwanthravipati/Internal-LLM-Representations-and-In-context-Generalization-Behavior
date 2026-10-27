import numpy as np
import pandas as pd
import pytest

from icl_geometry.analysis import robustness as rb


@pytest.mark.parametrize("args, expected", [
    ((0.3, 0.1, 0.5, 0.0, 1), "holds"),
    ((0.3, -0.1, 0.5, 0.0, 1), "weakens"),
    ((-0.3, -0.5, -0.1, 0.0, 1), "fails"),
    ((-0.3, -0.5, 0.1, 0.0, 1), "weakens"),
    ((0.4, 0.3, 0.45, 0.5, -1), "holds"),
    ((0.5, 0.4, 0.6, 0.5, 1), "weakens"),
    ((np.nan, 0.1, 0.5, 0.0, 1), "n/a"),
])
def test_classify_cell(args, expected):
    assert rb.classify_cell(*args) == expected


def test_classify_cell_min_n():
    assert rb.classify_cell(0.3, 0.1, 0.5, 0.0, 1, n=5, min_n=10) == "n/a"


def synthetic_exp2(seed=0, n_per_cell=10, d=40, gain=3.0):
    """One domain, cued attribute A (axis 0), uncued B (axis 1). Main-condition prompts with
    label_map 0 move the query toward its A class by `gain`; label_map 1 and the random-label
    control do not. Correctness on incongruent main prompts follows the shift for map 0 and is
    a coin flip for map 1."""
    rng = np.random.default_rng(seed)
    a = np.repeat([0, 0, 1, 1], n_per_cell)
    b = np.tile(np.repeat([0, 1], n_per_cell), 2)
    neutral = rng.standard_normal((len(a), d))
    neutral[:, 0] += 2 * (a - 0.5)
    neutral[:, 1] += 2 * (b - 0.5)
    split = np.where(np.arange(len(a)) % 2 == 0, "val", "test")
    rows, X, Xn = [], [], []
    for i in range(len(a)):
        for cond in ("main", "random_label"):
            for lm in (0, 1):
                for order in range(3):
                    shift = rng.normal(gain, 1.0) if (cond == "main" and lm == 0) else 0.0
                    v = neutral[i] + 0.3 * rng.standard_normal(d)
                    v[0] += shift * (a[i] - 0.5)
                    if lm == 0:
                        correct = shift + rng.normal(0, 0.5) > gain
                    else:
                        correct = rng.random() < 0.5
                    rows.append({"prompt_id": f"{i}-{cond}-{lm}-{order}", "domain": "d",
                                 "cued_attr": "A", "uncued_attr": "B", "m": 4,
                                 "condition": cond, "label_map": lm, "order": order,
                                 "query_item": f"item{i}",
                                 "query_attrs": {"A": int(a[i]), "B": int(b[i])},
                                 "congruent": bool(a[i] == b[i]), "correct": bool(correct),
                                 "cued_value": int(a[i]), "split": split[i]})
                    X.append(v)
                    Xn.append(neutral[i])
    return pd.DataFrame(rows), np.array(X), np.array(Xn)


def test_h2_permutation_null_vs_planted():
    df, X, _ = synthetic_exp2()
    obs = rb.h2_claims(X, df)["h2_delta_snr_cued"]
    null = rb.delta_snr_permutation_null(X, df, n_perm=30, seed=0)
    assert obs > np.percentile(null, 95)


def test_robustness_table_planted_label_map_effect():
    df, X, Xn = synthetic_exp2()
    fit, ev = (df.split == "val").to_numpy(), (df.split == "test").to_numpy()
    tr = rb.transform_variants(X[fit & (df.condition == "main").to_numpy()], k_drop=2)
    t = rb.robustness_table(df, X, X, Xn, fit, ev, transforms=tr, n_boot=100, min_n=8)
    cell = t.set_index(["claim", "variation", "level"]).cell
    assert cell["h2_delta_snr_cued", "all", "all"] == "holds"
    assert cell["h2_delta_snr_cued", "label_map", "0"] == "holds"
    assert cell["h2_delta_snr_cued", "label_map", "1"] == "weakens"
    assert cell["h3_shift_auroc", "label_map", "0"] == "holds"
    assert cell["h3_shift_auroc", "label_map", "1"] == "weakens"
    assert cell["h2_delta_snr_cued", "transform", "standardized"] == "holds"
    # deterministic
    t2 = rb.robustness_table(df, X, X, Xn, fit, ev, transforms=tr, n_boot=100, min_n=8)
    pd.testing.assert_frame_equal(t, t2)
    wide = rb.pivot_cells(t)
    assert set(wide.index) == set(rb.CLAIMS)
