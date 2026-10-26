---
id: T-021
title: Neutral representation protocol & representation sanity checks
status: in-progress
phase: 2
priority: P0
depends_on: [T-020, T-011]
owner: claude
estimate: S
tags: [geometry, design, science]
updated: 2026-10-26
---

# T-021 — Neutral representation protocol & representation sanity checks

## Context
Exp 1 applies the Sorscher theory to **context-free** item representations: what the model "thinks" of an item with no demonstrations. That needs a protocol. This ticket also settles how to handle LLM outlier ("massive activation") dimensions, which can dominate Euclidean geometry (PLAN §5).

## Tasks
- [ ] Define K ≥ 3 neutral carrier templates, e.g. `"{item}"` (after BOS), `"The word is {item}"`, `"This is about the {item}"`. Read the item's last subtoken.
- [ ] Decide the aggregation, **D-011**. Default: average across templates. Alternative: treat each template as an extra manifold sample. Report how sensitive Exp 1 results are to this choice.
- [ ] Extract neutral reps for all items, all layers, on the main models (via T-020).
- [ ] Sanity checks per layer:
  - cross-validated linear-probe category decodability;
  - RSA against the category structure;
  - participation ratio of each category manifold vs sample size P (shows bias at small P);
  - norms per dimension, to find outlier dimensions.
- [ ] Decide the outlier-dimension handling, **D-012**. Options: none; mean-centering; removing the top-k outlier dimensions; per-dimension standardization. **Note that whitening changes the theory's quantities.** Pick a primary handling in advance and keep the others as sensitivity analyses.

## Acceptance criteria
- [ ] Neutral reps cached for all items and main models.
- [ ] Sanity plots in `results/neutral_reps/REPORT.md`.
- [ ] D-011 and D-012 logged. Status → `review` (the student signs off on the protocol choices).

## Outputs
- `src/icl_geometry/models/neutral.py` (or `tasks/neutral_templates.py`), cache files, the report, DECISIONS entries.

## Notes & pitfalls
- A bare word right after BOS can behave oddly (attention sinks, first-token effects). That's one reason to use several templates.
- Keep the test split untouched: sanity checks can use all items for decodability, but don't choose layers here.

## References
- PLAN §5; `docs/background/sorscher_theory.md` (where the assumptions may break)

## Work log
- 2026-10-26 (claude): `src/icl_geometry/models/neutral.py` (4 carriers, last-subtoken states), `scripts/run_neutral.py` (cache `data/cache/<model>/neutral_<hash>/states.npy` [items, carriers, 25, d] + index + meta), `scripts/neutral_sanity.py`. Ran on **Qwen3.5-0.8B**: 1194 unique items, 209 s on CPU → `data/cache/qwen3.5-0.8b/neutral_bafed9d107e9/`; report `results/neutral_reps/qwen3.5-0.8b/REPORT.md`.
  - Findings (0.8B, 905 high-confidence Exp 1 items, 31 categories; chance 0.03): category decodability (5-fold CV, 128-PC logistic) 0.53 at the embeddings → 0.94 at layer 4 → ~0.90 through layer 24; nearest-centroid 0.51 → 0.90 (layer 4–6), with a dip to 0.77 around layers 12–15. RSA with category structure peaks at layer 5 (ρ = 0.25). Mean participation ratio of category manifolds is 14–22 and falls with depth to layer 14. D vs P at the middle layer shows the expected small-sample bias (see `D_vs_P.csv`). Outlier dims: top-1 variance share ≤ 0.04 at every layer, so no massive-activation problem at item tokens.
  - D-011 and D-012 written (proposed). 2B-Base neutral reps still to run (queued after the pilot).
