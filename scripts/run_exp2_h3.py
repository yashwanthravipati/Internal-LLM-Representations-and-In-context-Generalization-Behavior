"""T-033: does the per-prompt shift score predict generalization (H3)?

Needs two scored Exp 2 runs with cached `query_item` (and `decision`) activations: one on the
val split, one on the test split, plus the neutral cache for the same model.

    uv run python scripts/run_exp2_h3.py --val-run results/<val run> --test-run results/<test run>

Procedure (PLAN §6.4, §6.6, §6.9 item 2):
1. For each (domain, cued attribute) and layer, fit the cued axis on val main-condition
   in-context `query_item` reps (difference of class means).
2. Shift score for every prompt = signed projection of (in-context - neutral) on that axis.
3. Select the layer by val AUROC of the shift score for incongruent correctness.
4. On test: AUROC of {baseline}, {shift}, {baseline + shift} fitted on val, with CIs by
   bootstrap over query items, and the incremental gain of the shift score. Baseline = label
   bias (content-free margin) + m. Circular upper reference (reported separately): projection
   of the last-layer `decision` state on its own cued axis.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from icl_geometry import io
from icl_geometry.analysis.exp2 import load_run, neutral_lookup, shift_feature
from icl_geometry.analysis.prediction import evaluate, incremental
from icl_geometry.plotting import INK_MUTED, SERIES, apply_style, plt, save


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--layer", type=int, default=None, help="skip selection, use this layer")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    cfg_v, dv, av = load_run(args.val_run)
    cfg_t, dt, at = load_run(args.test_run)
    assert cfg_v.model == cfg_t.model, "val and test runs must use the same model"
    dv["split"], dt["split"] = "val", "test"
    neutral, row_of = neutral_lookup(cfg_v.model)
    n_states = av["query_item"].shape[1]

    out_cfg = io.RunConfig(exp="exp2h3", model=cfg_v.model, prompt_set=cfg_v.prompt_set,
                           desc="h3", extra=vars(args))
    with io.Run(out_cfg) as run:
        # 1-3: layer profile on val (axes fit on val main; evaluated on val incongruent main)
        prof = []
        for layer in range(n_states):
            s = shift_feature(dv, av["query_item"], neutral, row_of, layer,
                              np.ones(len(dv), bool))
            sel = (dv.condition == "main") & ~dv.congruent & np.isfinite(s)
            auc = roc_auc_score(dv.correct[sel], s[sel]) if dv.correct[sel].nunique() > 1 \
                else np.nan
            prof.append({"layer": layer, "val_auroc_shift": auc})
        prof = pd.DataFrame(prof)
        layer = args.layer if args.layer is not None else int(
            prof.loc[prof.val_auroc_shift.idxmax(), "layer"])

        # 4: features on both splits at the chosen layer (axes from val only)
        both = pd.concat([dv, dt], ignore_index=True)
        acts_q = np.concatenate([av["query_item"], at["query_item"]])
        acts_d = np.concatenate([av["decision"], at["decision"]])
        fit_mask = (both.split == "val").to_numpy()
        both["shift_score"] = shift_feature(both, acts_q, neutral, row_of, layer, fit_mask)
        both["circular_decision_proj"] = shift_feature(both, acts_d, neutral, row_of,
                                                       n_states - 1, fit_mask)
        both["label_bias"] = both.get("cf_margin", 0.0)
        both["m_f"] = both.m.astype(float)
        data = both[(both.condition == "main") & ~both.congruent].copy()
        data["correct"] = data.correct.astype(int)
        sets = {"baseline (label bias + m)": ["label_bias", "m_f"],
                "shift score": ["shift_score"],
                "baseline + shift": ["label_bias", "m_f", "shift_score"],
                "CIRCULAR: last-layer decision projection": ["circular_decision_proj"]}
        res = evaluate(data, sets, "correct", n_boot=args.n_boot)
        inc = incremental(data, ["label_bias", "m_f"], ["shift_score"], "correct",
                          n_boot=args.n_boot)
        io.write_parquet(both.drop(columns=["demos"]), run.path("features.parquet"))
        prof.to_csv(run.path("layer_profile_val.csv"), index=False)
        res.to_csv(run.path("h3_results.csv"), index=False)
        summary = {"selected_layer": layer, "incremental_shift": inc,
                   "results": res.to_dict("records")}
        run.path("summary.json").write_text(json.dumps(summary, indent=2, default=float))

        apply_style()
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(100 * prof.layer / (n_states - 1), prof.val_auroc_shift, color=SERIES[0])
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=":")
        ax.axvline(100 * layer / (n_states - 1), color=INK_MUTED, lw=1)
        ax.set(xlabel="Normalized depth", ylabel="Val AUROC (incongruent correct)",
               title="H3: shift score by layer (val; selection only)")
        save(fig, run.path("h3_layer_profile.png"))
        lines = [f"# Experiment 2, H3: {cfg_v.model}", "",
                 f"Val run `{args.val_run}`, test run `{args.test_run}`. Layer {layer} "
                 "selected on val; numbers below are on **test** (main condition, incongruent "
                 "queries). The circular row is an upper reference, not a predictor claim.", "",
                 res.round(3).to_markdown(index=False), "",
                 f"Incremental AUROC of the shift score over the baseline: {inc['gain']:.3f} "
                 f"(95% CI {inc['ci_low']:.3f} to {inc['ci_high']:.3f}).", "",
                 "![layer profile](h3_layer_profile.png)"]
        run.path("H3_REPORT.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
