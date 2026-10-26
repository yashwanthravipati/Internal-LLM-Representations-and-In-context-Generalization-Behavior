---
id: T-032
title: "Layer-wise H2 test: representational shift vs readout selection"
status: in-progress
phase: 3
priority: P1
depends_on: [T-031]
owner: claude
estimate: M
tags: [analysis, exp2, milestone, science]
updated: 2026-10-26
---

# T-032 — Layer-wise H2 test: representational shift vs readout selection

## Context
This is the central neuroscience question of Exp 2 (PLAN §2, §6.5):
- **H2a (Flesch-like):** the query item's own representation shifts toward the cued attribute, and the uncued attribute is compressed.
- **H2b (Mante-like):** both attributes stay decodable at the query item; selection happens only at the decision position.

## Tasks
- [ ] For each layer × position (`query_item`, `decision`) × condition: cued and uncued decodability, ΔSNR_cued, and RSA partial correlations. Report everything **relative to the random-label control**.
- [ ] Key figure: metrics vs normalized depth, with panels for `query_item` vs `decision` and lines for cued vs uncued, plus control bands.
- [ ] Interpret against the prediction table in PLAN §6.5. Give a verdict: H2a / H2b / mixed (say which layers and positions) / inconclusive, with statistics (bootstrap over items).
- [ ] Do it per domain as well as pooled. Domains may differ (semantic vs surface attributes).
- [ ] Write `results/exp2/H2_REPORT.md`.

## Acceptance criteria
- [ ] Report with the key figure and an explicit verdict.
- [ ] Analysis reproducible from a script.
- [ ] Status → `review` (interpretation needs the student).
- [ ] Milestone M3 is reached together with T-033.

## Outputs
- `src/icl_geometry/analysis/exp2_h2.py`, `results/exp2/H2_REPORT.md`, `figures/exp2_h2_*`.

## Notes & pitfalls
- Link the result to the task/function-vector literature: decision-position selection with item-position stability would fit ICL implemented as a readout "task vector".

## References
- PLAN §2, §6.5; Mante et al. 2013; Flesch et al. 2022; `docs/background/neuro_connections.md`

## Work log
- 2026-10-26 (claude): `src/icl_geometry/analysis/exp2.py` (`cell_metrics`, `layer_profile`: ΔSNR cued/uncued, decoding (prototype decoder, item-grouped CV, control task), CCGP, all vs the random-label control) and `scripts/run_exp2_h2.py` (per domain × cue × m × position × layer → `h2_metrics.parquet`, key figure, `H2_REPORT.md`). Smoke-tested end to end on the tiny 0.8B set. Fixed along the way: run-ID collisions within a minute (io.Run now suffixes `-2`), and degenerate manifolds (identical embeddings at layer 0) now give NaN instead of crashing. Not run on real data.
