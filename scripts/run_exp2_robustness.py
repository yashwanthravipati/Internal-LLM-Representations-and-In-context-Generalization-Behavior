"""T-034: robustness of the Exp 2 claims (H2 ΔSNR, H3 shift-score AUROC) across incidental
prompt variables and alternative analysis choices.

    uv run python scripts/run_exp2_robustness.py --val-run results/<val> --test-run results/<test>

Needs the same inputs as `run_exp2_h3.py` (two scored Exp 2 runs with cached `query_item`
activations, plus the neutral cache). Layers are selected on val unless given:
* H2 layer: max mean ΔSNR_cued (query_item) on val;
* H3 layer: max val AUROC of the shift score (same rule as run_exp2_h3.py).
Claims are evaluated on test. Writes ROBUSTNESS.md (the T-034 table; copy or link it as
results/exp2/ROBUSTNESS.md when reporting), robustness.csv and robustness_wide.csv into a new
run directory. Rules for holds / weakens / fails: `icl_geometry.analysis.robustness`.

Analysis variations: neutral aggregation (D-011: each single carrier vs the carrier mean) and
outlier handling (D-012: drop top-5 dimensions by mean |activation|, standardization).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.analysis import robustness as rb
from icl_geometry.analysis.exp2 import load_run, neutral_lookup, shift_feature_from


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--h2-layer", type=int, default=None)
    ap.add_argument("--h3-layer", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=200, help="H2 permutation null (all data)")
    ap.add_argument("--min-n", type=int, default=16, help="min prompts per cell, else n/a")
    ap.add_argument("--desc", default="robustness")
    args = ap.parse_args()
    cfg_v, dv, av = load_run(args.val_run)
    cfg_t, dt, at = load_run(args.test_run)
    assert cfg_v.model == cfg_t.model, "val and test runs must use the same model"
    dv["split"], dt["split"] = "val", "test"
    df = pd.concat([dv, dt], ignore_index=True)
    acts = np.concatenate([av["query_item"], at["query_item"]])
    neutral, row_of = neutral_lookup(cfg_v.model)
    n_states = acts.shape[1]
    fit = (df.split == "val").to_numpy()
    ev = (df.split == "test").to_numpy()
    val = df[fit].reset_index(drop=True)

    def X_at(layer):
        return acts[:, layer].astype(np.float64)

    # layer selection on val only
    h2_layer, h3_layer = args.h2_layer, args.h3_layer
    if h2_layer is None:
        prof = [rb.h2_claims(X_at(s)[fit], val)["h2_delta_snr_cued"] for s in range(n_states)]
        h2_layer = int(np.nanargmax(prof))
    if h3_layer is None:
        prof = []
        for s in range(n_states):
            Xn = rb.neutral_variants(df, neutral, row_of, s)["carrier_mean"]
            score = shift_feature_from(df, X_at(s), Xn, fit)
            prof.append(rb.h3_bootstrap(val, score[fit], n_boot=0).estimate.iloc[0])
        h3_layer = int(np.nanargmax(prof))

    out_cfg = io.RunConfig(exp="exp2robust", model=cfg_v.model, prompt_set=cfg_v.prompt_set,
                           desc=args.desc, extra=vars(args) | {"h2_layer": h2_layer,
                                                               "h3_layer": h3_layer})
    with io.Run(out_cfg) as run:
        X2, X3 = X_at(h2_layer), X_at(h3_layer)
        alts = rb.neutral_variants(df, neutral, row_of, h3_layer)
        Xn = alts.pop("carrier_mean")
        transforms = rb.transform_variants(X3[fit & (df.condition == "main").to_numpy()])
        transforms.pop("raw")
        table = rb.robustness_table(df, X2, X3, Xn, fit, ev, neutral_alts=alts,
                                    transforms=transforms, n_boot=args.n_boot,
                                    min_n=args.min_n)
        test = df[ev].reset_index(drop=True)
        obs = rb.h2_claims(X2[ev], test)["h2_delta_snr_cued"]
        null = rb.delta_snr_permutation_null(X2[ev], test, n_perm=args.n_perm)
        null = null[np.isfinite(null)]
        p_perm = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if len(null) else np.nan
        wide = rb.pivot_cells(table)
        table.to_csv(run.path("robustness.csv"), index=False)
        wide.to_csv(run.path("robustness_wide.csv"))
        summary = {"h2_layer": h2_layer, "h3_layer": h3_layer, "n_states": n_states,
                   "h2_delta_snr_cued_test": obs, "h2_perm_p": p_perm,
                   "cell_counts": table.cell.value_counts().to_dict()}
        run.path("summary.json").write_text(json.dumps(summary, indent=2, default=float))

        smoke = ["**SMOKE TEST: plumbing check only, not a result.**", ""] \
            if "smoke" in args.desc else []
        lines = [f"# Experiment 2 robustness (T-034): {cfg_v.model}", "", *smoke,
                 f"Val run `{args.val_run}` (layer selection, shift axes), test run "
                 f"`{args.test_run}` (all estimates). H2 at state {h2_layer}, H3 at state "
                 f"{h3_layer} (of {n_states}; 0 = embeddings), `query_item` position.", "",
                 "Cells: **holds** = same side of the null as the full-data estimate and the 95% "
                 "CI (bootstrap over query items) excludes the null; **weakens** = same side, CI "
                 "includes the null; **fails** = opposite side; **n/a** = too few prompts "
                 f"(< {args.min_n}) or no CI. Nulls: ΔSNR 0, AUROC 0.5.", "",
                 f"H2 ΔSNR_cued on test = {obs:.3f}; permutation p (condition labels shuffled "
                 f"within groups, {len(null)} perms) = {p_perm:.3f}.", "",
                 "## Table", "", wide.T.to_markdown(), "",
                 "## Estimates and CIs", "", table.round(3).to_markdown(index=False)]
        run.path("ROBUSTNESS.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:12]), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
