"""T-042 + T-043 (H5): does geometry in J-space predict generalization better than full hidden
states, against dimension-matched baselines?

Inputs: val and test Exp 2 runs with cached `query_item` activations (same model), the
neutral cache, and the model's lens. For each selected lens layer, every prompt's state (and
each query item's neutral state) is mapped into these representations:

  full          raw residual h (d dims)
  jspace        J-space component: non-negative gradient-pursuit reconstruction of h from
                <= k concept vectors (restricted concept vocabulary; d dims, sparse support)
  attr_span     coordinates in the span of all attribute-word J-lens vectors (r dims)
  logit_span    same span with the logit lens (J = I), the lens-free baseline (r dims)
  random_r      random orthonormal r-dim subspaces (mean over seeds), the dimension-matched null
  pca_r         top-r PCs of val in-context states, the unsupervised matched baseline

Metrics per representation (test split; anything fitted uses val only):
  * AUROC of the per-prompt shift score (axis fit on val) for incongruent correctness, CI by
    bootstrap over query items;
  * mean ΔSNR_cued (main vs random-label) across (domain, cued attribute, m) groups.
Also logged: J-space variance explained and the participation ratio of the J-space component.

Compute: pursuit costs ~N x T x k x d per layer. Defaults (3 middle layers, 8k-token
vocabulary, k = 25) suit a GPU; on CPU use --max-prompts for a smoke run.

    uv run python scripts/run_exp3_jspace.py --val-run results/<val> --test-run results/<test>
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from icl_geometry import io
from icl_geometry.analysis.exp2 import (
    attr_values,
    load_run,
    neutral_lookup,
    neutral_matrix,
    shift_feature_from,
)
from icl_geometry.analysis.prediction import evaluate
from icl_geometry.geometry.shift import snr_partition
from icl_geometry.geometry.sorscher import manifold_stats
from icl_geometry.geometry.subspaces import pca_basis, random_basis
from icl_geometry.lens.jlens_wrapper import Lens, concept_vectors, state_index
from icl_geometry.lens.jspace import (
    build_dictionary,
    concept_vocabulary,
    gradient_pursuit,
    span_projection,
    variance_explained,
)
from icl_geometry.lens.readout import attribute_word_ids
from icl_geometry.models.loading import load_model
from icl_geometry.tasks.prompts import load_domains


def delta_snr_groups(df: pd.DataFrame, X: np.ndarray) -> float:
    vals = []
    for _, g in df.groupby(["domain", "cued_attr", "m"]):
        gm, gc = g[g.condition == "main"], g[g.condition == "random_label"]
        if len(gm) < 8 or len(gc) < 8:
            continue
        m = float(g.m.iloc[0])
        vals.append(snr_partition(X[gm.index], gm.cued_value.to_numpy(), m)
                    - snr_partition(X[gc.index], gc.cued_value.to_numpy(), m))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-run", required=True)
    ap.add_argument("--test-run", required=True)
    ap.add_argument("--layers", default=None, help="comma-separated lens layers (default: "
                    "3 evenly spaced in the middle third)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--vocab-size", type=int, default=8000)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--max-prompts", type=int, default=None, help="per split (smoke runs)")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    cfg_v, dv, av = load_run(args.val_run)
    cfg_t, dt, at = load_run(args.test_run)
    assert cfg_v.model == cfg_t.model
    if args.max_prompts:
        dv = dv.sample(min(args.max_prompts, len(dv)), random_state=0).sort_index()
        dt = dt.sample(min(args.max_prompts, len(dt)), random_state=0).sort_index()
    dv["split"], dt["split"] = "val", "test"
    acts = np.concatenate([np.asarray(av["query_item"][dv.row.to_numpy()]),
                           np.asarray(at["query_item"][dt.row.to_numpy()])])
    df = pd.concat([dv, dt], ignore_index=True)
    df["uncued_value"] = attr_values(df, "uncued_attr")
    fit_mask = (df.split == "val").to_numpy()
    neutral, row_of = neutral_lookup(cfg_v.model)
    domains = load_domains()

    out_cfg = io.RunConfig(exp="exp3jspace", model=cfg_v.model, prompt_set=cfg_v.prompt_set,
                           desc="h5", extra=vars(args))
    with io.Run(out_cfg) as run:
        lm = load_model(cfg_v.model)
        lens = Lens.load(cfg_v.model)
        identity = Lens.logit_lens(cfg_v.model, lm.n_layers, lm.d_model)
        n = lm.n_layers
        layers = ([int(x) for x in args.layers.split(",")] if args.layers
                  else [round(n / 3 + i * n / 9) for i in range(3)])
        attr_ids = sorted({t for d in domains["exp2"]
                           for ids in attribute_word_ids(lm.tokenizer, domains, d).values()
                           for t in ids})
        vocab = concept_vocabulary(lm.tokenizer, required=attr_ids, max_size=args.vocab_size)

        rows, feats = [], []
        for layer in layers:
            s = state_index(layer)
            H = torch.from_numpy(acts[:, s].astype(np.float32))
            Hn = torch.from_numpy(neutral_matrix(df, neutral, row_of, s).astype(np.float32))
            dic = build_dictionary(lm, lens, layer, vocab)
            code, code_n = gradient_pursuit(H, dic.atoms, args.k), gradient_pursuit(
                Hn, dic.atoms, args.k)
            ve = variance_explained(H, code.recon)
            D_j = manifold_stats(code.recon.numpy()[fit_mask]).D
            span_j = concept_vectors(lm, lens, attr_ids, layer)
            span_l = concept_vectors(lm, identity, attr_ids, layer)
            reps = {
                "full": (H.numpy(), Hn.numpy()),
                "jspace": (code.recon.numpy(), code_n.recon.numpy()),
                "attr_span": (span_projection(H, span_j).numpy(),
                              span_projection(Hn, span_j).numpy()),
                "logit_span": (span_projection(H, span_l).numpy(),
                               span_projection(Hn, span_l).numpy()),
            }
            r = reps["attr_span"][0].shape[1]
            B = pca_basis(H.numpy()[fit_mask & (df.condition == "main").to_numpy()], r)
            reps["pca_r"] = (H.numpy() @ B, Hn.numpy() @ B)
            for seed in range(args.n_random):
                Br = random_basis(lm.d_model, r, seed)
                reps[f"random_r_{seed}"] = (H.numpy() @ Br, Hn.numpy() @ Br)

            layer_feats = pd.DataFrame({"prompt_id": df.prompt_id, "layer": layer})
            for name, (X, Xn) in reps.items():
                X, Xn = X.astype(np.float64), Xn.astype(np.float64)
                layer_feats[f"shift_{name}"] = shift_feature_from(df, X, Xn, fit_mask)
                test = df[df.split == "test"]
                rows.append({"layer": layer, "representation": name, "dim": X.shape[1],
                             "delta_snr_cued_test": delta_snr_groups(
                                 test.reset_index(drop=True), X[test.index.to_numpy()])})
            rows.append({"layer": layer, "representation": "_jspace_stats",
                         "variance_explained": ve, "D_jspace": D_j, "rank_attr_span": r})
            feats.append(layer_feats)

        geo = pd.DataFrame(rows)
        feats = pd.concat(feats, ignore_index=True)
        io.write_parquet(feats, run.path("shift_features.parquet"))

        # held-out AUROC per layer x representation (main, incongruent)
        res = []
        base = df[["prompt_id", "split", "query_item", "condition", "congruent", "correct"]]
        for layer, g in feats.groupby("layer"):
            data = base.merge(g, on="prompt_id")
            data = data[(data.condition == "main") & ~data.congruent].copy()
            data["correct"] = data.correct.astype(int)
            cols = [c for c in g.columns if c.startswith("shift_")]
            r_ = evaluate(data, {c.removeprefix("shift_"): [c] for c in cols}, "correct",
                          n_boot=args.n_boot)
            r_["layer"] = layer
            res.append(r_)
        res = pd.concat(res, ignore_index=True)
        rand = res[res.predictors.str.startswith("random_r_")]
        rand_summary = rand.groupby("layer").value.agg(["mean", "std"]).rename(
            columns={"mean": "random_r_mean", "std": "random_r_sd"})
        geo.to_csv(run.path("geometry.csv"), index=False)
        res.to_csv(run.path("h5_auroc.csv"), index=False)
        lines = [f"# H5: J-space vs full states vs matched baselines ({cfg_v.model})", "",
                 f"Val `{args.val_run}` (fitting), test `{args.test_run}` (evaluation); "
                 f"lens layers {layers}; k = {args.k}; vocabulary {len(vocab)} tokens.", "",
                 "## Test AUROC of shift score (main, incongruent)", "",
                 res[~res.predictors.str.startswith("random_r_")].round(3)
                 .to_markdown(index=False), "", "Random r-dim subspaces (mean ± sd over seeds):",
                 "", rand_summary.round(3).to_markdown(), "",
                 "## Geometry (ΔSNR_cued on test) and J-space statistics", "",
                 geo.round(4).to_markdown(index=False)]
        run.path("H5_REPORT.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
