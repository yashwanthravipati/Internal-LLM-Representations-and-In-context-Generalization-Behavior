"""T-021 sanity checks on neutral representations (Exp 1 items, all splits; no layer choice
is made here).

Per layer: cross-validated category decodability (logistic regression on 128 PCs; nearest-
centroid), RSA (Spearman between cosine-distance RDM and the same/different-category model
RDM), mean participation ratio of category manifolds and its dependence on P, and outlier-
dimension statistics. Writes results/neutral_reps/<model>/REPORT.md and figures.

    uv run python scripts/neutral_sanity.py --model Qwen/Qwen3.5-2B-Base
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from icl_geometry import io
from icl_geometry.geometry.sorscher import manifold_stats
from icl_geometry.plotting import SERIES, apply_style, plt, save
from icl_geometry.tasks.prompts import load_domains


def load_neutral(model: str):
    base = io.CACHE_DIR / io.model_short_name(model)
    dirs = sorted(base.glob("neutral_*"))
    dirs = [d for d in dirs if not d.name.endswith(".partial")]
    if not dirs:
        raise FileNotFoundError(f"no neutral cache under {base}; run scripts/run_neutral.py")
    d = dirs[-1]
    states = np.load(d / "states.npy", mmap_mode="r")
    index = pd.read_parquet(d / "index.parquet")
    return states, index, d


def outlier_stats(X: np.ndarray) -> dict:
    """X [n, d]: how much a few dimensions dominate."""
    var = X.var(0)
    mean_abs = np.abs(X.mean(0))
    share = np.sort(var)[::-1] / var.sum()
    return {
        "top1_var_share": float(share[0]),
        "top5_var_share": float(share[:5].sum()),
        "max_mean_abs_over_median": float(mean_abs.max() / np.median(mean_abs)),
        "top_dims_by_mean_abs": [int(i) for i in np.argsort(mean_abs)[::-1][:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    short = io.model_short_name(args.model)
    states, index, cache = load_neutral(args.model)
    domains = load_domains()
    cats = pd.read_csv(io.REPO_ROOT / domains["exp1"]["items"], dtype={"item": str})
    cats = cats[cats.confidence == "high"]
    row = dict(zip(index.item, index.row, strict=True))
    cats = cats[cats.item.isin(row)].reset_index(drop=True)
    rows_idx = np.array([row[i] for i in cats.item])
    y = cats.category.to_numpy()
    n_layers = states.shape[2]
    out_dir = io.REPO_ROOT / "results" / "neutral_reps" / short
    out_dir.mkdir(parents=True, exist_ok=True)

    model_rdm = pdist(pd.get_dummies(y).to_numpy().astype(float), "hamming") > 0
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    records = []
    for layer in range(n_layers):
        X = states[rows_idx, :, layer].astype(np.float64).mean(1)  # mean over carriers
        rec = {"layer": layer}
        logreg = make_pipeline(StandardScaler(), PCA(128, random_state=0),
                               LogisticRegression(max_iter=2000, C=1.0))
        rec["probe_acc"] = float(cross_val_score(logreg, X, y, cv=cv).mean())
        rec["centroid_acc"] = float(cross_val_score(NearestCentroid(), X, y, cv=cv).mean())
        rec["rsa_rho"] = float(spearmanr(pdist(X, "cosine"), model_rdm).statistic)
        Ds = [manifold_stats(X[y == c]).D for c in np.unique(y)]
        rec["mean_D"] = float(np.mean(Ds))
        rec["norm_mean"] = float(np.linalg.norm(X, axis=1).mean())
        rec |= outlier_stats(X)
        # carrier sensitivity: mean cosine between per-carrier reps of the same item
        per = states[rows_idx, :, layer].astype(np.float64)
        per = per / np.linalg.norm(per, axis=-1, keepdims=True)
        k = per.shape[1]
        cos = [(per[:, i] * per[:, j]).sum(-1).mean() for i in range(k) for j in range(i + 1, k)]
        rec["carrier_cos"] = float(np.mean(cos))
        records.append(rec)
        print(f"L{layer:2d} probe={rec['probe_acc']:.2f} centroid={rec['centroid_acc']:.2f} "
              f"rsa={rec['rsa_rho']:.2f} D={rec['mean_D']:.1f} "
              f"top1var={rec['top1_var_share']:.2f} carrier_cos={rec['carrier_cos']:.2f}")
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "layer_profile.csv", index=False)

    # D vs P at a middle layer (sample-size bias of the participation ratio)
    mid = n_layers // 2
    X = states[rows_idx, :, mid].astype(np.float64).mean(1)
    rng = np.random.default_rng(0)
    dp = []
    for c in np.unique(y):
        Xc = X[y == c]
        for P in (5, 10, 15, 20, len(Xc)):
            if P <= len(Xc):
                vals = [manifold_stats(Xc[rng.choice(len(Xc), P, replace=False)]).D
                        for _ in range(20)]
                dp.append({"category": c, "P": P if P < len(Xc) else "all",
                           "P_n": P, "D": float(np.mean(vals))})
    dp = pd.DataFrame(dp)
    dp.to_csv(out_dir / "D_vs_P.csv", index=False)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(df.layer, df.probe_acc, color=SERIES[0], label="Linear probe (128 PCs)")
    axes[0].plot(df.layer, df.centroid_acc, color=SERIES[1], label="Nearest centroid")
    axes[0].axhline(1 / len(np.unique(y)), color="#8a8984", lw=1, ls="--")
    axes[0].set(title="Category decodability (5-fold CV)", xlabel="Layer (0 = embeddings)",
                ylabel="Accuracy", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(df.layer, df.rsa_rho, color=SERIES[0])
    axes[1].set(title="RSA with category structure", xlabel="Layer", ylabel="Spearman ρ")
    axes[2].plot(df.layer, df.top1_var_share, color=SERIES[0], label="Top-1 dim variance share")
    axes[2].plot(df.layer, df.carrier_cos, color=SERIES[1], label="Carrier agreement (cos)")
    axes[2].set(title="Outlier dimension & carrier agreement", xlabel="Layer", ylim=(0, 1))
    axes[2].legend()
    fig.suptitle(f"Neutral representations, {args.model}", x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out_dir / "layer_profile.png")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    g = dp.groupby("P_n").D.mean()
    ax.plot(g.index, g.values, marker="o", color=SERIES[0])
    ax.set(title=f"Participation ratio vs samples (layer {mid})", xlabel="P (items)",
           ylabel="Mean D across categories")
    save(fig, out_dir / "D_vs_P.png")

    lines = [f"# Neutral representation sanity checks: {args.model}", "",
             f"Cache: `{cache.relative_to(io.REPO_ROOT)}`; {len(cats)} high-confidence Exp 1 "
             f"items, {len(np.unique(y))} categories; representation = mean over "
             f"{states.shape[1]} carriers.", "",
             "![layer profile](layer_profile.png)", "", "![D vs P](D_vs_P.png)", "",
             df.round(3).to_markdown(index=False), "",
             "## D vs P (middle layer)", "",
             dp.groupby("P_n").D.agg(["mean", "std"]).round(2).to_markdown(), ""]
    (out_dir / "REPORT.md").write_text("\n".join(lines))
    (out_dir / "summary.json").write_text(json.dumps({
        "best_probe_layer": int(df.probe_acc.idxmax()),
        "best_probe_acc": float(df.probe_acc.max()),
        "max_top1_var_share": float(df.top1_var_share.max()),
    }, indent=2))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
