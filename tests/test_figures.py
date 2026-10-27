"""scripts/make_figures.py renders every figure from synthetic run directories with the same
files and columns the analysis scripts write (no model, no real results needed)."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from icl_geometry import io

spec = importlib.util.spec_from_file_location("make_figures",
                                              io.REPO_ROOT / "scripts" / "make_figures.py")
mf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mf)

MODEL = "Qwen/Qwen3.5-0.8B"
N_STATES = 25


def _run(root, name, extra=None):
    d = root / name
    d.mkdir()
    cfg = io.RunConfig(exp=name, model=MODEL, prompt_set="p", extra=extra or {})
    (d / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    (d / "metadata.json").write_text(json.dumps({"run_id": name, "git": {"commit": "abc"},
                                                 "status": "ok"}))
    return d


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("results")
    rng = np.random.default_rng(0)
    out = {}

    d = _run(root, "exp1", {"eval_split": "test", "select_split": "val"})
    layers = np.arange(N_STATES)
    pd.DataFrame([{"layer": int(l_), "split": s, "rho_theory": rng.uniform(),
                   "rho_signal_only": rng.uniform(), "rho_proto": rng.uniform(),
                   "rho_theory_vs_proto": rng.uniform()}
                  for s in ("val", "test") for l_ in layers]).to_csv(d / "layer_profile.csv",
                                                                       index=False)
    tasks = pd.DataFrame([{"a": f"c{i}", "b": f"c{j}", "m": m, "layer": int(l_), "split": s,
                           "theory_error": rng.uniform(0, 0.5), "error": rng.uniform(0, 0.5),
                           "proto_error": rng.uniform(0, 0.5), "signal": rng.uniform()}
                          for i in range(4) for j in range(4) if i != j for m in (1, 2, 4, 8)
                          for l_ in (3, 4) for s in ("val", "test")])
    tasks.to_parquet(d / "tasks.parquet")
    (d / "summary.json").write_text(json.dumps({
        "selected_layer": 4, "n_layers_states": N_STATES, "rho": 0.4, "ci95": [0.1, 0.6],
        "null_95pct": 0.2, "error_measure": "error"}))
    out["exp1"] = d

    d = _run(root, "t022")
    pd.DataFrame([{"kind": k, "m": m, "theory_full": rng.uniform(), "mc_error": rng.uniform()}
                  for k in ("high_d", "low_d") for m in (1, 2) for _ in range(5)]
                 ).to_parquet(d / "theory_vs_mc.parquet")
    out["t022"] = d

    d = _run(root, "t023")
    pd.DataFrame([{"layer": l_, "m": m, "theory_error": rng.uniform(0, 0.5),
                   "proto_error": rng.uniform(0, 0.5)}
                  for l_ in (0, 12, 24) for m in (1, 4) for _ in range(5)]
                 ).to_parquet(d / "theory_vs_mc.parquet")
    out["t023"] = d

    d = _run(root, "h2")
    h2 = pd.DataFrame([{"position": p, "layer": l_, "depth": 100 * l_ / (N_STATES - 1),
                        "domain": "animals", "cued_attr": c, "m": 4,
                        **{k: rng.normal() for k in ("delta_snr_cued", "delta_snr_uncued",
                                                     "delta_acc_cued", "delta_acc_uncued")}}
                       for p in ("query_item", "decision") for l_ in (0, 8, 16, 24)
                       for c in ("class", "habitat")])
    h2.to_parquet(d / "h2_metrics.parquet")
    out["h2"] = d

    d = _run(root, "h3")
    pd.DataFrame([{"predictors": p, "metric": "auroc", "value": 0.6, "ci_low": 0.5,
                   "ci_high": 0.7, "n_fit": 10, "n_eval": 10}
                  for p in ("baseline (label bias + m)", "shift score", "baseline + shift",
                            "CIRCULAR: last-layer decision projection")]
                 ).to_csv(d / "h3_results.csv", index=False)
    pd.DataFrame({"layer": layers, "val_auroc_shift": rng.uniform(0.4, 0.7, N_STATES)}
                 ).to_csv(d / "layer_profile_val.csv", index=False)
    (d / "summary.json").write_text(json.dumps({"selected_layer": 10, "incremental_shift": {
        "gain": 0.05, "ci_low": -0.01, "ci_high": 0.1}}))
    out["h3"] = d

    for name in ("jlens", "logit"):
        d = _run(root, name)
        pd.DataFrame([{"position": p, "layer": l_, "condition": c,
                       "rule_readout": rng.normal()}
                      for p in ("query_item", "decision") for l_ in (0, 11, 23)
                      for c in ("random_label", "main", "explicit_rule")]
                     ).to_parquet(d / "readout.parquet")
        out[name] = d

    d = _run(root, "h5")
    reps = ["full", "jspace", "attr_span", "logit_span", "pca_r", "random_r_0", "random_r_1"]
    pd.DataFrame([{"predictors": r, "metric": "auroc", "value": rng.uniform(0.4, 0.7),
                   "ci_low": 0.3, "ci_high": 0.8, "n_fit": 5, "n_eval": 5, "layer": l_}
                  for l_ in (8, 11) for r in reps]).to_csv(d / "h5_auroc.csv", index=False)
    out["h5"] = d
    return out


RUN_KEYS = {"exp1_rho_by_layer": ["exp1"], "exp1_scatter": ["exp1"],
            "exp1_learning_curves": ["exp1"], "theory_vs_prototype_synthetic": ["t022"],
            "theory_vs_prototype_neutral": ["t023"], "h2_profile": ["h2"],
            "h3_auroc": ["h3"], "h4_readout": ["jlens", "logit"], "h5_auroc": ["h5"]}


def test_every_figure_function_is_covered():
    assert set(RUN_KEYS) == set(mf.FIGURES)


@pytest.mark.parametrize("name", sorted(RUN_KEYS))
def test_figure_renders(name, runs, tmp_path):
    entry = {"name": name, "runs": {k: str(runs[k]) for k in RUN_KEYS[name]}, "status": "final"}
    path = mf.make_one(entry, fig_dir=tmp_path)
    assert path.exists() and path.with_suffix(".pdf").exists()
    src = json.loads((tmp_path / f"{name}.source.json").read_text())
    assert all(s["git_commit"] == "abc" for s in src["sources"].values())


def test_smoke_figures_are_quarantined(runs, tmp_path):
    entry = {"name": "h2_profile", "runs": {"h2": str(runs["h2"])}, "status": "smoke"}
    path = mf.make_one(entry, fig_dir=tmp_path)
    assert path.parent == tmp_path / "smoke"


def test_pending_and_bad_entries(runs, tmp_path):
    assert mf.make_one({"name": "h2_profile", "runs": {"h2": "results/<todo>"}}) is None
    assert mf.make_one({"name": "h2_profile", "runs": None}) is None
    with pytest.raises(ValueError):
        mf.make_one({"name": "h2_profile", "runs": {"h2": str(runs["h2"])}, "status": "draft"},
                    fig_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        mf.make_one({"name": "h2_profile", "runs": {"h2": str(runs["h2"]) + "_missing"}},
                    fig_dir=tmp_path)


def test_manifests_reference_known_functions():
    for name in ("manifest.yaml", "manifest_smoke.yaml"):
        entries = yaml.safe_load((io.REPO_ROOT / "figures" / name).read_text())["figures"]
        for e in entries:
            assert e.get("fn", e["name"]) in mf.FIGURES, e["name"]
            assert e.get("status", "final") in ("final", "exploratory", "smoke")
