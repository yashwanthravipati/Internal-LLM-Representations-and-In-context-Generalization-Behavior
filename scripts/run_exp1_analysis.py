"""T-024: Experiment 1, Sorscher theory from neutral reps vs LLM few-shot error.

    uv run python scripts/run_exp1_analysis.py --model Qwen/Qwen3.5-2B-Base \
        --behavior results/<run_id> [--select-split val --eval-split test]

With only validation behavior available (the pilot), run with `--select-split val
--eval-split val`; the report is then labelled EXPLORATORY (layer chosen and evaluated on the
same split; no headline claims).

Outputs go to a new run directory: tasks.parquet (per layer x task), layer_profile.csv,
REPORT.md and figures.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.analysis.exp1 import (
    bootstrap_over_categories,
    llm_task_errors,
    permutation_null,
    spearman,
    theory_table,
)
from icl_geometry.plotting import INK_MUTED, SERIES, apply_style, plt, save
from icl_geometry.tasks.prompts import PromptConfig, exp1_items, load_domains


def load_behavior(run_dirs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    beh, prompts = [], []
    for d in run_dirs:
        d = io.REPO_ROOT / d
        cfg = io.RunConfig.from_yaml(d / "config.yaml")
        beh.append(pd.read_parquet(d / "behavior.parquet"))
        prompts.append(pd.DataFrame(io.read_jsonl(io.REPO_ROOT / cfg.prompt_set)))
    return pd.concat(prompts, ignore_index=True), pd.concat(beh, ignore_index=True)


def neutral_reps(model: str):
    base = io.CACHE_DIR / io.model_short_name(model)
    d = sorted(p for p in base.glob("neutral_*") if not p.name.endswith(".partial"))[-1]
    states = np.load(d / "states.npy", mmap_mode="r")
    index = pd.read_parquet(d / "index.parquet")
    return states, dict(zip(index.item, index.row, strict=True)), d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--behavior", nargs="+", required=True, help="run dirs with behavior")
    ap.add_argument("--select-split", default="val")
    ap.add_argument("--eval-split", default="test")
    ap.add_argument("--error", default="error", choices=["error", "error_cal"])
    ap.add_argument("--n-draws", type=int, default=200, help="Monte Carlo prototype draws")
    ap.add_argument("--geometry-items", default="split", choices=["split", "all"],
                    help="estimate manifolds from the prompt split's items or all items")
    ap.add_argument("--desc", default=None, help="run-id suffix (default: analysis or "
                    "exploratory); use 'smoke' for plumbing checks")
    args = ap.parse_args()
    exploratory = args.select_split == args.eval_split

    prompts, beh = load_behavior(args.behavior)
    prompts = prompts[prompts.experiment == "exp1"]
    states, row_of, cache = neutral_reps(args.model)
    domains = load_domains()
    items = exp1_items(PromptConfig(name="x", experiment="exp1", split="val"), domains)

    cfg = io.RunConfig(exp="exp1", model=args.model, prompt_set=",".join(args.behavior),
                       desc=args.desc or ("exploratory" if exploratory else "analysis"),
                       extra=vars(args) | {"neutral_cache": str(cache)})
    with io.Run(cfg) as run:
        results, profiles = {}, []
        for split in sorted({args.select_split, args.eval_split}):
            p = prompts[prompts.split == split]
            tasks = llm_task_errors(p, beh)
            if tasks.empty:
                raise SystemExit(f"no exp1 behavior for split {split}")
            cats = sorted(set(tasks.a) | set(tasks.b))
            pool = items if args.geometry_items == "all" else items[items.split == split]
            per_layer = []
            for layer in range(states.shape[2]):
                reps = {}
                for c in cats:
                    its = pool[pool.category == c].item
                    reps[c] = states[[row_of[i] for i in its], :, layer].astype(np.float64).mean(1)
                th = theory_table(reps, tasks, n_draws=args.n_draws if split == args.eval_split
                                  else 0)
                th = th.merge(tasks, on=["a", "b", "m"])
                th["layer"] = layer
                per_layer.append(th)
            df = pd.concat(per_layer, ignore_index=True)
            df["split"] = split
            results[split] = df
            prof = df.groupby("layer").apply(
                lambda g: pd.Series({
                    "rho_theory": spearman(g.theory_error, g[args.error]),
                    "rho_signal_only": spearman(-g.signal, g[args.error]),
                    "rho_proto": spearman(g.proto_error, g[args.error]) if "proto_error" in g
                    else np.nan,
                    "rho_theory_vs_proto": spearman(g.theory_error, g.proto_error)
                    if "proto_error" in g else np.nan,
                }), include_groups=False).reset_index()
            prof["split"] = split
            profiles.append(prof)
        profile = pd.concat(profiles, ignore_index=True)
        profile.to_csv(run.path("layer_profile.csv"), index=False)
        all_tasks = pd.concat(results.values(), ignore_index=True)
        io.write_parquet(all_tasks, run.path("tasks.parquet"))

        sel = profile[profile.split == args.select_split]
        best = int(sel.loc[sel.rho_theory.idxmax(), "layer"])
        ev = results[args.eval_split]
        evb = ev[ev.layer == best].reset_index(drop=True)
        rho = spearman(evb.theory_error, evb[args.error])
        ci = bootstrap_over_categories(evb, "theory_error", args.error)
        null = permutation_null(evb.theory_error, evb[args.error])
        p_perm = float((np.sum(null >= rho) + 1) / (len(null) + 1))
        rho_sig = spearman(-evb.signal, evb[args.error])
        rho_l0 = spearman(ev[ev.layer == 0].theory_error, ev[ev.layer == 0][args.error])
        summary = {"selected_layer": best, "n_layers_states": int(states.shape[2]),
                   "rho": rho, "ci95": ci, "perm_p": p_perm,
                   "null_95pct": float(np.percentile(null, 95)),
                   "rho_signal_only": rho_sig, "rho_layer0": rho_l0,
                   "n_tasks": len(evb), "exploratory": exploratory,
                   "error_measure": args.error}
        if "proto_error" in evb:
            summary["rho_proto_vs_llm"] = spearman(evb.proto_error, evb[args.error])
            summary["rho_theory_vs_proto"] = spearman(evb.theory_error, evb.proto_error)
        run.path("summary.json").write_text(json.dumps(summary, indent=2))

        # ---------------- figures ----------------
        apply_style()
        n_states = states.shape[2]
        depth = lambda layer: 100 * layer / (n_states - 1)  # noqa: E731
        fig, ax = plt.subplots(figsize=(5.5, 3.3))
        for color, split in zip(SERIES, sorted(results), strict=False):
            pr = profile[profile.split == split]
            ax.plot(depth(pr.layer), pr.rho_theory, color=color, label=f"Theory ({split})")
            ax.plot(depth(pr.layer), pr.rho_signal_only, color=color, ls="--",
                    label=f"Signal only ({split})")
        ax.axhline(summary["null_95pct"], color=INK_MUTED, lw=1, ls=":",
                   label="Permutation null 95%")
        ax.axvline(depth(best), color=INK_MUTED, lw=1)
        ax.set(xlabel="Normalized depth (0 = embeddings)", ylabel="Spearman ρ with LLM error",
               title="Exp 1: theory vs LLM error by layer")
        ax.legend(fontsize=7)
        save(fig, run.path("exp1_rho_by_layer.png"))

        fig, ax = plt.subplots(figsize=(4.5, 4))
        for color, m in zip(SERIES, sorted(evb.m.unique()), strict=False):
            sub = evb[evb.m == m]
            ax.scatter(sub.theory_error, sub[args.error], s=12, color=color, alpha=0.75,
                       edgecolors="white", linewidths=0.4, label=f"m = {m}")
        ax.plot([0, 0.6], [0, 0.6], color=INK_MUTED, lw=1, ls="--")
        ax.set(xlabel=f"Theory error (layer {best})", ylabel="LLM error",
               title=f"Exp 1 ({args.eval_split}): ρ = {rho:.2f}")
        ax.legend(fontsize=7)
        save(fig, run.path("exp1_scatter.png"))

        fig, ax = plt.subplots(figsize=(4.5, 3.3))
        lc = evb.groupby("m")[[args.error, "theory_error"]
                              + (["proto_error"] if "proto_error" in evb else [])].mean()
        ax.plot(lc.index, lc[args.error], marker="o", color=SERIES[0], label="LLM")
        ax.plot(lc.index, lc.theory_error, marker="o", color=SERIES[1], label="Theory")
        if "proto_error" in lc:
            ax.plot(lc.index, lc.proto_error, marker="o", color=SERIES[2],
                    label="Prototype simulation")
        ax.set(xscale="log", xticks=[1, 2, 4, 8], xticklabels=["1", "2", "4", "8"],
               xlabel="m (demos per category)", ylabel="Mean error", title="Learning curves")
        ax.legend(fontsize=7)
        save(fig, run.path("exp1_learning_curves.png"))

        tag = "EXPLORATORY (layer selected and evaluated on the same split)" if exploratory \
            else f"layer selected on {args.select_split}, evaluated on {args.eval_split}"
        lines = [f"# Experiment 1: {args.model}", "", f"**{tag}.**", "",
                 f"Behavior runs: {', '.join(f'`{b}`' for b in args.behavior)}; neutral cache "
                 f"`{cache.relative_to(io.REPO_ROOT)}`; geometry items: {args.geometry_items}; "
                 f"error: {args.error}.", "",
                 "```json", json.dumps(summary, indent=2), "```", "",
                 "![rho by layer](exp1_rho_by_layer.png)", "", "![scatter](exp1_scatter.png)",
                 "", "![learning curves](exp1_learning_curves.png)", "",
                 "## Layer profile", "", profile.round(3).to_markdown(index=False)]
        run.path("REPORT.md").write_text("\n".join(lines) + "\n")
        print(json.dumps(summary, indent=2), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
