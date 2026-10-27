"""T-060: regenerate every report figure from saved run artifacts (no model needed).

    uv run python scripts/make_figures.py                                # figures/manifest.yaml
    uv run python scripts/make_figures.py --manifest figures/manifest_smoke.yaml
    uv run python scripts/make_figures.py --only exp1_scatter

Each manifest entry names a figure function below, the run directories it reads, and a
status:

* `final`: goes to figures/<name>.png + .pdf.
* `exploratory`: same place, with an EXPLORATORY corner label.
* `smoke`: goes to figures/smoke/ (gitignored) with a SMOKE TEST watermark. Never report these.

Entries whose runs are not filled in yet (`runs: null` or a path containing `<`) are listed as
PENDING and skipped. Every figure gets a `<name>.source.json` sidecar with the source run IDs,
their git commits and the manifest entry, so each figure traces back to `results/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from icl_geometry import io
from icl_geometry.plotting import (
    INK_MUTED,
    ROLE_COLORS,
    SERIES,
    apply_style,
    banner,
    depth,
    plt,
    save,
    watermark,
)

FIG_DIR = io.REPO_ROOT / "figures"
# Transformer blocks per model; residual states = blocks + 1 (index 0 = embeddings).
MODEL_LAYERS = {"Qwen/Qwen3.5-0.8B": 24, "Qwen/Qwen3.5-2B-Base": 24}
CONDITION_ORDER = ["main", "random_label", "explicit_rule", "ambiguous"]


def _model(run: Path) -> str:
    return io.RunConfig.from_yaml(run / "config.yaml").model


def _n_states(run: Path) -> int:
    return MODEL_LAYERS[_model(run)] + 1


def _ci_errorbar(ax, x, df, color, label, marker="o"):
    ax.errorbar(x, df.value, yerr=[df.value - df.ci_low, df.ci_high - df.value], fmt=marker,
                color=color, ms=6, capsize=0, elinewidth=1.5, label=label)


# ---------------------------------------------------------------------------
# Experiment 1 (H1)
# ---------------------------------------------------------------------------


def exp1_rho_by_layer(runs):
    run = runs["exp1"]
    prof = pd.read_csv(run / "layer_profile.csv")
    summ = json.loads((run / "summary.json").read_text())
    eval_split = io.RunConfig.from_yaml(run / "config.yaml").extra["eval_split"]
    n = summ["n_layers_states"]
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    for split in ("val", "test"):
        pr = prof[prof.split == split]
        if pr.empty:
            continue
        alpha = 1.0 if split == eval_split else 0.45  # the selection split is drawn faded
        ax.plot(depth(pr.layer, n), pr.rho_theory, color=ROLE_COLORS["theory"],
                alpha=alpha, label=f"Theory ({split})")
        ax.plot(depth(pr.layer, n), pr.rho_signal_only, color=INK_MUTED, ls="--",
                alpha=alpha, label=f"Centroid distance only ({split})")
        if pr.rho_proto.notna().any():
            ax.plot(depth(pr.layer, n), pr.rho_proto, color=ROLE_COLORS["prototype"],
                    alpha=alpha, label=f"Prototype simulation ({split})")
    ax.axhline(summ["null_95pct"], color=INK_MUTED, lw=1, ls=":", label="Permutation null 95%")
    ax.axvline(depth(summ["selected_layer"], n), color=INK_MUTED, lw=1)
    ax.set(xlabel="Normalized depth (0 = embeddings)", ylabel="Spearman ρ with LLM error",
           title="Exp 1: geometry predicts few-shot error, by layer")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def _exp1_eval(run: Path) -> tuple[pd.DataFrame, dict]:
    summ = json.loads((run / "summary.json").read_text())
    cfg = io.RunConfig.from_yaml(run / "config.yaml")
    tasks = pd.read_parquet(run / "tasks.parquet")
    ev = tasks[(tasks.split == cfg.extra["eval_split"]) & (tasks.layer == summ["selected_layer"])]
    return ev, summ | {"eval_split": cfg.extra["eval_split"], "error_col": summ["error_measure"]}


def exp1_scatter(runs):
    ev, summ = _exp1_eval(runs["exp1"])
    err = summ["error_col"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ms = sorted(ev.m.unique())
    for color, m in zip(SERIES, ms, strict=False):
        sub = ev[ev.m == m]
        ax.scatter(sub.theory_error, sub[err], s=14, color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.5, label=f"m = {m}")
    hi = float(np.nanmax([ev.theory_error.max(), ev[err].max(), 0.5]))
    ax.plot([0, hi], [0, hi], color=INK_MUTED, lw=1, ls="--", zorder=0)
    lo_ci, hi_ci = summ["ci95"]
    ax.set(xlabel=f"Theory error (layer {summ['selected_layer']})", ylabel="LLM error",
           title=f"Exp 1 ({summ['eval_split']}): ρ = {summ['rho']:.2f} "
                 f"[{lo_ci:.2f}, {hi_ci:.2f}]")
    ax.legend(fontsize=7, title="Demos per category", title_fontsize=7)
    return fig


def exp1_learning_curves(runs):
    ev, summ = _exp1_eval(runs["exp1"])
    err = summ["error_col"]
    cols = [err, "theory_error"] + (["proto_error"] if "proto_error" in ev else [])
    lc = ev.groupby("m")[cols].mean()
    fig, ax = plt.subplots(figsize=(4.5, 3.3))
    ax.plot(lc.index, lc[err], marker="o", color=ROLE_COLORS["llm"], label="LLM")
    ax.plot(lc.index, lc.theory_error, marker="o", color=ROLE_COLORS["theory"], label="Theory")
    if "proto_error" in lc:
        ax.plot(lc.index, lc.proto_error, marker="o", color=ROLE_COLORS["prototype"],
                label="Prototype simulation")
    ax.set(xscale="log", xticks=list(lc.index), xticklabels=[str(m) for m in lc.index],
           xlabel="m (demos per category)", ylabel="Mean error",
           title="Learning curves (mean over tasks)")
    ax.minorticks_off()
    ax.legend(fontsize=7)
    return fig


# ---------------------------------------------------------------------------
# Theory validation (T-022 / T-023)
# ---------------------------------------------------------------------------


def theory_vs_prototype_synthetic(runs):
    df = pd.read_parquet(runs["t022"] / "theory_vs_mc.parquet")
    ms = sorted(df.m.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(2.75 * len(ms), 3.1), sharex=True,
                             sharey=True)
    labels = {"high_d": "High-D manifolds (flat spectrum)",
              "low_d": "Low-D manifolds (power law)"}
    for ax, m in zip(np.atleast_1d(axes), ms, strict=True):
        for color, kind in zip(SERIES[:2], ("high_d", "low_d"), strict=True):
            sub = df[(df.m == m) & (df.kind == kind)]
            ax.scatter(sub.theory_full, sub.mc_error, s=14, color=color, alpha=0.8,
                       edgecolors="white", linewidths=0.5, label=labels[kind])
        ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="--", zorder=0)
        ax.set(title=f"m = {m}", xlabel="Theory error (Eq. 1, full)")
    np.atleast_1d(axes)[0].set_ylabel("Simulated prototype error")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.suptitle("Sorscher theory vs Monte Carlo prototype learning (synthetic manifolds)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def theory_vs_prototype_neutral(runs):
    run = runs["t023"]
    df = pd.read_parquet(run / "theory_vs_mc.parquet")
    layers = sorted(df.layer.unique())
    n = _n_states(run)
    fig, axes = plt.subplots(1, len(layers), figsize=(2.9 * len(layers), 3.1), sharex=True,
                             sharey=True)
    ms = sorted(df.m.unique())
    for ax, layer in zip(np.atleast_1d(axes), layers, strict=True):
        for color, m in zip(SERIES, ms, strict=False):
            sub = df[(df.layer == layer) & (df.m == m)]
            ax.scatter(sub.theory_error, sub.proto_error, s=10, color=color, alpha=0.7,
                       edgecolors="white", linewidths=0.4, label=f"m = {m}")
        ax.plot([0, 0.6], [0, 0.6], color=INK_MUTED, lw=1, ls="--", zorder=0)
        ax.set(title=f"Depth {depth(layer, n):.0f} (state {layer})",
               xlabel="Theory error (Eq. 1, full)")
    np.atleast_1d(axes)[0].set_ylabel("Simulated prototype error")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(f"Theory vs prototype simulation on neutral item representations "
                 f"({_model(run)})", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment 2 (H2, H3)
# ---------------------------------------------------------------------------


def h2_profile(runs):
    df = pd.read_parquet(runs["h2"] / "h2_metrics.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for col, position in enumerate(("query_item", "decision")):
        g = df[df.position == position].groupby("depth")
        for row, (mc, mu, ylabel) in enumerate((
                ("delta_snr_cued", "delta_snr_uncued", "ΔSNR vs random-label control"),
                ("delta_acc_cued", "delta_acc_uncued", "Δ decoding accuracy vs control"))):
            ax = axes[row, col]
            for role, metric in (("cued", mc), ("uncued", mu)):
                mean = g[metric].mean()
                lo, hi = g[metric].quantile(0.25), g[metric].quantile(0.75)
                ax.plot(mean.index, mean.values, color=ROLE_COLORS[role],
                        label=f"{role} attribute")
                ax.fill_between(mean.index, lo.values, hi.values, color=ROLE_COLORS[role],
                                alpha=0.15, lw=0)
            ax.axhline(0, color=INK_MUTED, lw=1)
            ax.set(ylabel=ylabel, title=position.replace("_", " "))
            if row == 1:
                ax.set_xlabel("Normalized depth (0 = embeddings)")
            ax.legend(fontsize=7)
    fig.suptitle("Exp 2 (H2): shift (cued ΔSNR up, uncued down) vs selection (only the readout "
                 "changes); line = mean over domain × cue × m, band = IQR", x=0.01, ha="left",
                 fontsize=10)
    fig.tight_layout()
    return fig


def h3_auroc(runs):
    run = runs["h3"]
    res = pd.read_csv(run / "h3_results.csv")
    prof = pd.read_csv(run / "layer_profile_val.csv")
    summ = json.loads((run / "summary.json").read_text())
    n = _n_states(run)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    ax0.plot(depth(prof.layer, n), prof.val_auroc_shift, color=SERIES[0])
    ax0.axhline(0.5, color=INK_MUTED, lw=1, ls=":")
    ax0.axvline(depth(summ["selected_layer"], n), color=INK_MUTED, lw=1)
    ax0.set(xlabel="Normalized depth", ylabel="Val AUROC", title="Layer selection (val only)")
    y = np.arange(len(res))[::-1]
    circular = res.predictors.str.startswith("CIRCULAR")
    for yi, (_, r) in zip(y, res.iterrows(), strict=True):
        color = INK_MUTED if r.predictors.startswith("CIRCULAR") else SERIES[0]
        ax1.errorbar(r.value, yi, xerr=[[r.value - r.ci_low], [r.ci_high - r.value]], fmt="o",
                     color=color, ms=6, elinewidth=1.5)
    ax1.set_yticks(y, [p.replace("CIRCULAR: ", "circular ref.: ") for p in res.predictors],
                   fontsize=8)
    ax1.axvline(0.5, color=INK_MUTED, lw=1, ls=":")
    if circular.any():
        ax1.axhline(y[circular.to_numpy()].max() + 0.5, color=INK_MUTED, lw=0.6)
    inc = summ["incremental_shift"]
    ax1.set(xlabel="Test AUROC, incongruent queries (95% CI over items)",
            title=f"Held-out prediction; shift gain = {inc['gain']:.3f} "
                  f"[{inc['ci_low']:.3f}, {inc['ci_high']:.3f}]")
    ax1.grid(axis="y", visible=False)
    fig.suptitle("Exp 2 (H3): does the representational shift predict generalization?",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Experiment 3 (H4, H5)
# ---------------------------------------------------------------------------


def h4_readout(runs):
    """Rule readout by depth for the J-lens and (optionally) the logit-lens baseline."""
    lenses = [(k, runs[k]) for k in ("jlens", "logit") if k in runs]
    fig, axes = plt.subplots(len(lenses), 2, figsize=(10, 3.2 * len(lenses)), sharex=True,
                             sharey=True, squeeze=False)
    for i, (name, run) in enumerate(lenses):
        res = pd.read_parquet(run / "readout.parquet")
        n_blocks = MODEL_LAYERS[_model(run)]
        for j, pos in enumerate(("query_item", "decision")):
            ax = axes[i, j]
            sub = res[res.position == pos]
            conds = [c for c in CONDITION_ORDER if c in set(sub.condition)]
            for cond in conds + sorted(set(sub.condition) - set(conds)):
                mean = sub[sub.condition == cond].groupby("layer").rule_readout.mean()
                ax.plot(depth(mean.index + 1, n_blocks + 1), mean.values,
                        color=ROLE_COLORS.get(cond, SERIES[4]), label=cond.replace("_", " "))
            ax.axhline(0, color=INK_MUTED, lw=1)
            lens_name = "J-lens" if name == "jlens" else "logit lens (J = I)"
            ax.set(title=f"{lens_name}: {pos.replace('_', ' ')}",
                   ylabel="Rank cued − uncued words")
            if i == len(lenses) - 1:
                ax.set_xlabel("Normalized depth")
            ax.legend(fontsize=7)
    fig.suptitle("Exp 3 (H4): is the demonstrated rule poised for verbalization?", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


def h5_auroc(runs):
    run = runs["h5"]
    res = pd.read_csv(run / "h5_auroc.csv")
    n = _n_states(run)
    layers = sorted(res.layer.unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(max(5.5, 3.6 * len(layers)), 3.6),
                             sharey=True,
                             squeeze=False)
    names = ["full", "jspace", "attr_span", "logit_span", "pca_r"]
    for ax, layer in zip(axes[0], layers, strict=True):
        sub = res[res.layer == layer]
        rand = sub[sub.predictors.str.startswith("random_r_")].value
        if len(rand):
            mu, sd = rand.mean(), rand.std(ddof=1) if len(rand) > 1 else 0.0
            ax.axhspan(mu - sd, mu + sd, color=ROLE_COLORS["random_r"], alpha=0.2, lw=0,
                       label="random r-dim (mean ± sd)")
        for x, name in enumerate(names):
            r = sub[sub.predictors == name]
            if len(r):  # the x tick labels name the points; no legend entry needed
                _ci_errorbar(ax, [x], r, ROLE_COLORS[name], None)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(names)), [s.replace("_", "\n") for s in names], fontsize=7)
        ax.grid(axis="x", visible=False)
        ax.set(title=f"Lens layer {layer} (depth {depth(layer + 1, n):.0f})")
    axes[0, 0].set_ylabel("Test AUROC of shift score")
    axes[0, 0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.suptitle("Exp 3 (H5): J-space vs full states vs dimension-matched baselines", x=0.01,
                 ha="left")
    fig.tight_layout()
    return fig


FIGURES = {f.__name__: f for f in (
    exp1_rho_by_layer, exp1_scatter, exp1_learning_curves, theory_vs_prototype_synthetic,
    theory_vs_prototype_neutral, h2_profile, h3_auroc, h4_readout, h5_auroc)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolve_runs(entry: dict) -> dict[str, Path] | None:
    runs = entry.get("runs")
    if not runs or any(v is None or "<" in str(v) for v in runs.values()):
        return None
    return {k: io.REPO_ROOT / v for k, v in runs.items()}


def make_one(entry: dict, fig_dir: Path = FIG_DIR) -> Path | None:
    """Render one manifest entry; None if its runs are not filled in yet."""
    runs = _resolve_runs(entry)
    if runs is None:
        return None
    missing = [str(p) for p in runs.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{entry['name']}: missing run dirs {missing}")
    status = entry.get("status", "final")
    if status not in ("final", "exploratory", "smoke"):
        raise ValueError(f"{entry['name']}: unknown status {status!r}")
    apply_style()
    fig = FIGURES[entry.get("fn", entry["name"])](runs)
    out_dir = fig_dir / "smoke" if status == "smoke" else fig_dir
    if status == "smoke":
        watermark(fig, "SMOKE TEST: NOT A RESULT")
    elif status == "exploratory":
        banner(fig, "EXPLORATORY")
    path = save(fig, out_dir / f"{entry['name']}.png", formats=("pdf",))
    sources = {}
    for k, p in runs.items():
        meta = json.loads((p / "metadata.json").read_text())
        rel = p.relative_to(io.REPO_ROOT) if p.is_relative_to(io.REPO_ROOT) else p
        sources[k] = {"run_dir": str(rel), "run_id": meta.get("run_id"),
                      "git_commit": meta.get("git", {}).get("commit"),
                      "status": meta.get("status")}
    (out_dir / f"{entry['name']}.source.json").write_text(
        json.dumps({"figure": entry, "sources": sources}, indent=2, default=str))
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(FIG_DIR / "manifest.yaml"))
    ap.add_argument("--only", nargs="*", help="figure names to (re)make")
    args = ap.parse_args()
    entries = yaml.safe_load(Path(args.manifest).read_text())["figures"]
    made, pending = [], []
    for entry in entries:
        if args.only and entry["name"] not in args.only:
            continue
        path = make_one(entry)
        if path is None:
            pending.append(entry["name"])
            print(f"PENDING  {entry['name']} ({entry.get('ticket', '')}): runs not filled in")
        else:
            made.append(path)
            rel = path.relative_to(io.REPO_ROOT) if path.is_relative_to(io.REPO_ROOT) else path
            print(f"made     {rel} [{entry.get('status', 'final')}]")
    print(f"\n{len(made)} made, {len(pending)} pending")


if __name__ == "__main__":
    main()
