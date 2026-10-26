"""T-032: layer-wise H2 test (representational shift vs readout selection), relative to the
random-label control.

    uv run python scripts/run_exp2_h2.py --run results/<exp2 run with cached activations>

Writes a new results/<run_id>/ with h2_metrics.parquet, H2_REPORT.md and the key figure.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.analysis.exp2 import layer_profile
from icl_geometry.plotting import INK_MUTED, SERIES, apply_style, plt, save


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with behavior + activation cache")
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--group-by", default="domain,cued_attr,m")
    args = ap.parse_args()
    src = io.REPO_ROOT / args.run
    meta_run = json.loads((src / "metadata.json").read_text())
    cfg = io.RunConfig.from_yaml(src / "config.yaml")
    prompts = pd.DataFrame(io.read_jsonl(io.REPO_ROOT / cfg.prompt_set))
    beh = pd.read_parquet(src / "behavior.parquet")
    cache = io.REPO_ROOT / meta_run["activation_cache"]
    group_cols = tuple(args.group_by.split(","))

    out_cfg = io.RunConfig(exp="exp2h2", model=cfg.model, prompt_set=cfg.prompt_set,
                           desc="h2", extra={"source_run": args.run, "group_by": group_cols})
    with io.Run(out_cfg) as run:
        frames = []
        for position in ("query_item", "decision"):
            acts, index = io.load_activations(cache, position)
            meta = index.merge(prompts, on="prompt_id").merge(beh, on="prompt_id")
            meta = meta.sort_values("row").reset_index(drop=True)
            assert (meta.row.to_numpy() == np.arange(len(meta))).all()
            layers = list(range(0, acts.shape[1], args.layer_step))
            prof = layer_profile(acts, meta, layers, group_cols)
            prof["position"] = position
            frames.append(prof)
        df = pd.concat(frames, ignore_index=True)
        n_states = acts.shape[1]
        df["depth"] = 100 * df.layer / (n_states - 1)
        io.write_parquet(df, run.path("h2_metrics.parquet"))

        apply_style()
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
        for col, position in enumerate(("query_item", "decision")):
            sub = df[df.position == position]
            g = sub.groupby("depth")
            for row, (metric_c, metric_u, ylabel) in enumerate((
                    ("delta_snr_cued", "delta_snr_uncued", "ΔSNR vs random-label control"),
                    ("delta_acc_cued", "delta_acc_uncued", "Δ decoding accuracy vs control"))):
                ax = axes[row, col]
                for color, metric, label in ((SERIES[0], metric_c, "cued"),
                                             (SERIES[1], metric_u, "uncued")):
                    mean = g[metric].mean()
                    lo, hi = g[metric].quantile(0.25), g[metric].quantile(0.75)
                    ax.plot(mean.index, mean.values, color=color, label=f"{label} attribute")
                    ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.15,
                                    lw=0)
                ax.axhline(0, color=INK_MUTED, lw=1)
                ax.set(ylabel=ylabel, title=f"{position}")
                if row == 1:
                    ax.set_xlabel("Normalized depth (0 = embeddings)")
                ax.legend(fontsize=7)
        fig.suptitle("Exp 2 (H2): in-context vs random-label control; line = mean over "
                     "domain × cue × m, band = IQR", x=0.01, ha="left")
        fig.tight_layout()
        save(fig, run.path("h2_profile.png"))

        summ = (df.groupby(["position", "layer"])[
            ["delta_snr_cued", "delta_snr_uncued", "delta_acc_cued", "delta_acc_uncued",
             "acc_cued_main", "acc_uncued_main", "acc_cued_ctrl", "acc_uncued_ctrl"]]
            .mean().round(3))
        lines = [f"# Experiment 2, H2: {cfg.model}", "",
                 f"Source run `{args.run}`; grouped by {', '.join(group_cols)}.", "",
                 "![H2 profile](h2_profile.png)", "", "## Mean over groups", "",
                 summ.to_markdown(), ""]
        for dom, g in df.groupby("domain"):
            lines += [f"## {dom}", "", g.groupby(["position", "cued_attr", "layer"])[
                ["delta_snr_cued", "delta_snr_uncued", "delta_acc_cued", "delta_acc_uncued"]]
                .mean().round(3).to_markdown(), ""]
        run.path("H2_REPORT.md").write_text("\n".join(lines))
        print(summ.to_string(), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
