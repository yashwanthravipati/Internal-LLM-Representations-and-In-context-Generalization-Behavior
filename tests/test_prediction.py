import numpy as np
import pandas as pd

from icl_geometry.analysis.prediction import evaluate, incremental, mixed_effects


def synthetic(n_items=60, per_item=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        item_eff = rng.normal(0, 0.5)
        split = "val" if i < n_items // 2 else "test"
        for k in range(per_item):
            useful = rng.normal()
            noise_feat = rng.normal()
            base = rng.normal()
            margin = 1.5 * useful + 0.5 * base + item_eff + rng.normal(0, 0.5)
            rows.append({"query_item": f"it{i}", "split": split, "useful": useful,
                         "noise": noise_feat, "base": base, "margin": margin,
                         "correct": margin > 0, "demo_key": f"d{k % 3}"})
    return pd.DataFrame(rows)


def test_evaluate_ranks_predictors():
    df = synthetic()
    res = evaluate(df, {"useful": ["useful"], "noise": ["noise"]}, "correct", n_boot=200)
    r = res.set_index("predictors")
    assert r.loc["useful", "value"] > 0.8 and r.loc["useful", "ci_low"] > 0.7
    assert abs(r.loc["noise", "value"] - 0.5) < 0.1
    res2 = evaluate(df, {"useful": ["useful"]}, "margin", n_boot=200)
    assert res2.metric[0] == "r2" and res2.value[0] > 0.5


def test_incremental():
    df = synthetic()
    gain = incremental(df, ["base"], ["useful"], "correct", n_boot=200)
    assert gain["gain"] > 0.1 and gain["ci_low"] > 0
    null = incremental(df, ["useful"], ["noise"], "correct", n_boot=200)
    assert null["ci_low"] < 0 < null["ci_high"] or abs(null["gain"]) < 0.02


def test_mixed_effects_recovers_slope():
    df = synthetic()
    fit = mixed_effects(df, "margin ~ useful + base")
    assert abs(fit.params["useful"] - 1.5) < 0.15
