"""T-023: Sorscher theory vs Monte Carlo prototype (and exemplar) learners on cached neutral item
representations (analogue of paper Fig. 3A on LLM states). Reads the neutral activation cache
only; no model is run.

    uv run python scripts/validate_theory_neutral.py --model Qwen/Qwen3.5-2B-Base

This checks the math and its approximations on the real geometry (low D, few items per
category, heavy tails). It is not a test of H1: no LLM behavior is involved.
"""

from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
import pandas as pd

from icl_geometry import io
from icl_geometry.analysis.exp1 import spearman
from icl_geometry.analysis.exp2 import neutral_lookup
from icl_geometry.geometry.prototype import prototype_error
from icl_geometry.geometry.sorscher import manifold_stats, pair_terms, snr_from_terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default=None,
                    help="comma-separated state indices (default: 5 evenly spaced)")
    ap.add_argument("--m", default="1,2,4,8")
    ap.add_argument("--max-pairs", type=int, default=300, help="ordered category pairs")
    ap.add_argument("--n-draws", type=int, default=300)
    ap.add_argument("--min-items", type=int, default=12)
    ap.add_argument("--desc", default="theory-vs-mc")
    args = ap.parse_args()

    states, row_of = neutral_lookup(args.model)
    n_states = states.shape[2]
    layers =([int(x) for x in args.layers.split(",")] if args.layers
              else sorted({round(i * (n_states - 1) / 4) for i in range(5)}))
    m_grid = [int(x) for x in args.m.split(",")]
    items = pd.read_csv(io.REPO_ROOT / "data" / "items" / "categories.csv")
    items = items[items["item"].isin(row_of)]
    counts = items.category.value_counts()
    cats = sorted(counts[counts >= args.min_items].index)
    rng = np.random.default_rng(0)
    pairs = [(a, b) for a in cats for b in cats if a != b]
    if len(pairs) > args.max_pairs:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), args.max_pairs, replace=False))]

    cfg = io.RunConfig(exp="t023", model=args.model, prompt_set="none", desc=args.desc,
                       extra=vars(args) | {"layers": layers, "n_categories": len(cats),
                                           "n_pairs": len(pairs)})
    rows = []
    with io.Run(cfg) as run:
        for layer in layers:
            reps = {c: states[[row_of[i] for i in items[items.category == c]["item"]], :, layer]
                    .astype(np.float64).mean(1) for c in cats}
            stats = {c: manifold_stats(X) for c, X in reps.items()}
            for a, b in pairs:
                terms = pair_terms(stats[a], stats[b])
                for m in m_grid:
                    if m >= min(len(reps[a]), len(reps[b])):
                        continue
                    seed = zlib.crc32(f"{a}|{b}|{m}|{layer}".encode())
                    proto, _ = prototype_error(reps[a], reps[b], m, args.n_draws, seed)
                    exem, _ = prototype_error(reps[a], reps[b], m, args.n_draws, seed,
                                              rule="exemplar")
                    rows.append({
                        "layer": layer, "a": a, "b": b, "m": m, "P_a": terms["P_a"],
                        "D_a": terms["D_a"], "signal": terms["signal"],
                        "theory_error": float(snr_from_terms(terms, m, full=True)["error"]),
                        "theory_dominant": float(snr_from_terms(terms, m)["error"]),
                        "proto_error": proto, "exemplar_error": exem,
                    })
        df = pd.DataFrame(rows)
        io.write_parquet(df, run.path("theory_vs_mc.parquet"))
        df["abs_err"] = (df.theory_error - df.proto_error).abs()
        df["bias"] = df.theory_error - df.proto_error
        summary = df.groupby(["layer", "m"]).apply(lambda g: pd.Series({
            "n": len(g), "median_D": g.D_a.median(),
            "mae": g.abs_err.mean(), "mean_bias": g.bias.mean(),
            "rho_theory_proto": spearman(g.theory_error, g.proto_error),
            "rho_proto_exemplar": spearman(g.proto_error, g.exemplar_error),
        }), include_groups=False).reset_index()
        summary.to_csv(run.path("summary.csv"), index=False)
        # where do theory and simulation diverge? (rank correlation of |error| with D, P)
        diverge = {
            "rho_abs_err_vs_D": spearman(df.D_a, df.abs_err),
            "rho_abs_err_vs_P": spearman(df.P_a, df.abs_err),
            "rho_abs_err_vs_m": spearman(df.m, df.abs_err),
            "overall_mae": float(df.abs_err.mean()),
            "overall_mean_bias": float(df.bias.mean()),
        }
        run.path("summary.json").write_text(json.dumps(diverge, indent=2))
        lines = [f"# T-023: theory vs prototype simulation on neutral reps ({args.model})", "",
                 "Math/approximation check on real geometry; not a test of H1.", "",
                 f"{len(cats)} categories with ≥ {args.min_items} items, {len(pairs)} ordered "
                 f"pairs, layers {layers}, m ∈ {m_grid}, {args.n_draws} draws.", "",
                 "```json", json.dumps(diverge, indent=2), "```", "",
                 summary.round(3).to_markdown(index=False)]
        run.path("REPORT.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), "\nrun dir:", run.dir)


if __name__ == "__main__":
    main()
