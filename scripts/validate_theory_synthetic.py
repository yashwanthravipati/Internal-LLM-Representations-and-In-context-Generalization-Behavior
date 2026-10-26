"""T-022/T-023 validation: Sorscher theory vs Monte Carlo prototype learning on synthetic
Gaussian manifolds (analogue of paper Fig. 2E-H / 3A). Writes a figure and a CSV under a run
directory, and a copy of the figure to figures/.

    uv run python scripts/validate_theory_synthetic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.geometry.prototype import prototype_error
from icl_geometry.geometry.sorscher import manifold_stats, snr
from icl_geometry.plotting import INK_MUTED, SERIES, apply_style, plt, save

N_DIM = 200
N_SAMPLES = 1000
M_GRID = (1, 2, 5, 10)


def make_pair(rng, kind: str):
    """Random manifold pair. 'high_d': flat spectrum over 60-120 axes. 'low_d': power-law
    spectrum (D ~ 2-6). Random separations and radius ratios."""
    k = int(rng.integers(60, 121))
    if kind == "high_d":
        radii = np.ones(k)
    else:
        radii = 1.0 / np.arange(1, k + 1) ** rng.uniform(0.8, 1.2)
    radii = radii / np.sqrt((radii**2).sum())  # total variance 1
    ratio = rng.uniform(0.7, 1.4)
    ax_a = np.linalg.qr(rng.standard_normal((N_DIM, k)))[0]
    ax_b = np.linalg.qr(rng.standard_normal((N_DIM, k)))[0]
    sep = rng.uniform(0.2, 1.6)
    direction = rng.standard_normal(N_DIM)
    direction /= np.linalg.norm(direction)
    Xa = (rng.standard_normal((N_SAMPLES, k)) * radii) @ ax_a.T
    Xb = sep * direction + (rng.standard_normal((N_SAMPLES, k)) * radii * ratio) @ ax_b.T
    return Xa, Xb


def main() -> None:
    cfg = io.RunConfig(exp="t022", model="synthetic", prompt_set="none", desc="theory-vs-mc",
                       extra={"n_dim": N_DIM, "n_samples": N_SAMPLES, "m_grid": list(M_GRID),
                              "pairs_per_kind": 25, "n_draws": 500})
    rng = np.random.default_rng(cfg.seed)
    rows = []
    with io.Run(cfg) as run:
        for kind in ("high_d", "low_d"):
            for pair in range(cfg.extra["pairs_per_kind"]):
                Xa, Xb = make_pair(rng, kind)
                sa, sb = manifold_stats(Xa), manifold_stats(Xb)
                for m in M_GRID:
                    mc_a, mc_b = prototype_error(Xa, Xb, m, n_draws=cfg.extra["n_draws"],
                                                 rng=int(rng.integers(1 << 31)))
                    for d, mc, (s1, s2) in (("a", mc_a, (sa, sb)), ("b", mc_b, (sb, sa))):
                        rows.append({
                            "kind": kind, "pair": pair, "direction": d, "m": m, "D": s1.D,
                            "mc_error": mc,
                            "theory_dominant": float(snr(s1, s2, m)["error"]),
                            "theory_full": float(snr(s1, s2, m, full=True)["error"]),
                        })
        df = pd.DataFrame(rows)
        io.write_parquet(df, run.path("theory_vs_mc.parquet"))

        summary = (
            df.assign(abs_dom=(df.theory_dominant - df.mc_error).abs(),
                      abs_full=(df.theory_full - df.mc_error).abs())
            .groupby(["kind", "m"])
            .agg(median_D=("D", "median"), mae_dominant=("abs_dom", "mean"),
                 mae_full=("abs_full", "mean"),
                 r_full=("theory_full",
                         lambda s: np.corrcoef(s, df.loc[s.index, "mc_error"])[0, 1]))
            .reset_index()
        )
        summary.to_csv(run.path("summary.csv"), index=False)
        print(summary.to_string(index=False))

        apply_style()
        fig, axes = plt.subplots(1, len(M_GRID), figsize=(11, 3.1), sharex=True, sharey=True)
        labels = {"high_d": "High-D manifolds (flat spectrum)",
                  "low_d": "Low-D manifolds (power law)"}
        for ax, m in zip(axes, M_GRID, strict=True):
            for color, kind in zip(SERIES[:2], ("high_d", "low_d"), strict=True):
                sub = df[(df.m == m) & (df.kind == kind)]
                ax.scatter(sub.theory_full, sub.mc_error, s=14, color=color, alpha=0.8,
                           edgecolors="white", linewidths=0.5, label=labels[kind])
            ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--", zorder=0)
            ax.set_title(f"m = {m}")
            ax.set_xlabel("Theory error (Eq. 1, full)")
        axes[0].set_ylabel("Simulated prototype error")
        axes[0].legend(loc="upper left")
        fig.suptitle("Sorscher theory vs Monte Carlo prototype learning (synthetic manifolds)",
                     x=0.01, ha="left", fontsize=11)
        fig.tight_layout()
        save(fig, run.path("theory_vs_prototype_synthetic.png"))
        fig_copy = io.REPO_ROOT / "figures" / "theory_vs_prototype_synthetic.png"
        fig_copy.write_bytes(run.path("theory_vs_prototype_synthetic.png").read_bytes())
        print("run dir:", run.dir)


if __name__ == "__main__":
    main()
