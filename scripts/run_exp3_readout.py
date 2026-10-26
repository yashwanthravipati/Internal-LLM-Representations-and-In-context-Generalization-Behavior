"""T-041 (H4): is the demonstrated rule readable in the Jacobian lens?

For every prompt in a scored Exp 2 run with cached activations, computes the lens percentile
rank of every attribute word at `query_item` and `decision`, for each lens layer, then

    rule_readout = mean rank(cued words) - mean rank(uncued words)

and summarizes it by condition (main / random_label = negative control / explicit_rule =
positive control) and by correctness. `--logit-lens` swaps in J = I as the baseline.

    uv run python scripts/run_exp3_readout.py --run results/<exp2 run> [--limit 200]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from icl_geometry import io
from icl_geometry.lens.jlens_wrapper import Lens
from icl_geometry.lens.readout import attribute_word_ids, percentile_ranks
from icl_geometry.models.loading import load_model
from icl_geometry.plotting import INK_MUTED, SERIES, apply_style, plt, save
from icl_geometry.tasks.prompts import load_domains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--logit-lens", action="store_true")
    args = ap.parse_args()
    src = io.REPO_ROOT / args.run
    cfg = io.RunConfig.from_yaml(src / "config.yaml")
    meta = json.loads((src / "metadata.json").read_text())
    prompts = pd.DataFrame(io.read_jsonl(io.REPO_ROOT / cfg.prompt_set))
    beh = pd.read_parquet(src / "behavior.parquet")
    cache = io.REPO_ROOT / meta["activation_cache"]
    domains = load_domains()

    out_cfg = io.RunConfig(exp="exp3readout", model=cfg.model, prompt_set=cfg.prompt_set,
                           desc="logitlens" if args.logit_lens else "jlens",
                           extra=vars(args))
    with io.Run(out_cfg) as run:
        lm = load_model(cfg.model)
        lens = (Lens.logit_lens(cfg.model, lm.n_layers, lm.d_model) if args.logit_lens
                else Lens.load(cfg.model))
        layers = lens.layers[:: args.layer_step]
        rows = []
        for position in ("query_item", "decision"):
            acts, index = io.load_activations(cache, position)
            df = index.merge(prompts, on="prompt_id").merge(beh, on="prompt_id")
            if args.limit:
                df = df.sample(min(args.limit, len(df)), random_state=0)
            for domain, g in df.groupby("domain"):
                words = attribute_word_ids(lm.tokenizer, domains, domain)
                attrs = list(words)
                all_ids = words[attrs[0]] + words[attrs[1]]
                ranks = percentile_ranks(lm, lens, np.asarray(acts[g.row.to_numpy()]), layers,
                                         all_ids)
                n0 = len(words[attrs[0]])
                mean_rank = {attrs[0]: ranks[:, :, :n0].mean(-1),
                             attrs[1]: ranks[:, :, n0:].mean(-1)}
                for k, r in enumerate(g.itertuples()):
                    for j, layer in enumerate(layers):
                        rows.append({
                            "prompt_id": r.prompt_id, "domain": domain, "position": position,
                            "layer": layer, "condition": r.condition, "cued_attr": r.cued_attr,
                            "m": r.m, "congruent": r.congruent, "correct": r.correct,
                            "rank_cued": mean_rank[r.cued_attr][k, j],
                            "rank_uncued": mean_rank[r.uncued_attr][k, j],
                        })
        res = pd.DataFrame(rows)
        res["rule_readout"] = res.rank_cued - res.rank_uncued
        io.write_parquet(res, run.path("readout.parquet"))

        summ = (res.groupby(["position", "layer", "condition"]).rule_readout.mean()
                .unstack("condition").round(4))
        aucs = []
        for (pos, layer), g in res[(res.condition == "main") & ~res.congruent].groupby(
                ["position", "layer"]):
            if g.correct.nunique() > 1:
                aucs.append({"position": pos, "layer": layer,
                             "auroc_correct": roc_auc_score(g.correct, g.rule_readout)})
        aucs = pd.DataFrame(aucs)

        apply_style()
        n_layers = lm.n_layers
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
        for ax, pos in zip(axes, ("query_item", "decision"), strict=True):
            sub = summ.loc[pos]
            for color, cond in zip(SERIES, sub.columns, strict=False):
                ax.plot(100 * (sub.index + 1) / n_layers, sub[cond], color=color, label=cond)
            ax.axhline(0, color=INK_MUTED, lw=1)
            ax.set(title=pos, xlabel="Normalized depth",
                   ylabel="Rule readout (rank cued − uncued)")
            ax.legend(fontsize=7)
        fig.suptitle(f"H4 rule readout ({'logit lens' if args.logit_lens else 'J-lens'}), "
                     f"{cfg.model}", x=0.01, ha="left")
        fig.tight_layout()
        save(fig, run.path("h4_readout.png"))
        lens_name = "logit lens (J = I)" if args.logit_lens else "J-lens"
        lines = [f"# H4 rule readout: {cfg.model}", "",
                 f"Source `{args.run}`; lens: {lens_name}; {res.prompt_id.nunique()} prompts.",
                 "", "![readout](h4_readout.png)", "",
                 "## Mean rule readout by condition", "", summ.to_markdown(), "",
                 "## AUROC of rule readout for correctness (main, incongruent)", "",
                 aucs.round(3).to_markdown(index=False) if len(aucs) else "(no variance)"]
        run.path("H4_REPORT.md").write_text("\n".join(lines) + "\n")
        print(summ.to_string(), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
