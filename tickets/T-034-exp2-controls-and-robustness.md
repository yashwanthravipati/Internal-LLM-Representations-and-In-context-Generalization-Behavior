---
id: T-034
title: Controls & robustness for Experiment 2
status: in-progress
phase: 3
priority: P1
depends_on: [T-032, T-033]
owner: claude
estimate: M
tags: [analysis, exp2, controls]
updated: 2026-10-27
---

# T-034 — Controls & robustness for Experiment 2

## Context
Before putting Exp 2 claims in the report, check that they don't hinge on incidental choices.

## Tasks
- [ ] Re-run the key H2/H3 metrics across:
  - label sets and label→group mappings;
  - demo orders;
  - templates;
  - m;
  - outlier-dimension handling (D-012 alternatives);
  - neutral-template aggregation (D-011 alternatives).
- [ ] Repeat the headline results on a **second model size**.
- [ ] Summarize which results are robust, fragile or reversed.

## Acceptance criteria
- [ ] `results/exp2/ROBUSTNESS.md` with a robustness table (rows: claims; columns: variations; cells: holds / weakens / fails).

## Outputs
- `results/exp2/ROBUSTNESS.md`.

## Notes & pitfalls
- If time is short, prioritize: second model size > label mapping > outlier handling > the rest (PLAN §8 cut list).

## References
- PLAN §10

## Work log
- 2026-10-27 (claude): Code built ahead of the real Exp 2 runs (no model needed).
  - `src/icl_geometry/analysis/robustness.py`: `classify_cell` (rules in D-015), H2 claims + bootstrap, a permutation null for ΔSNR_cued (shuffle main/random_label within group), H3 bootstrap, D-012 transforms (raw / drop top-5 / standardize), D-011 neutral variants (carrier mean / single carriers), `robustness_table` + `pivot_cells`.
  - `scripts/run_exp2_robustness.py --val-run --test-run` → `ROBUSTNESS.md`, `robustness.csv`, `robustness_wide.csv`, `summary.json` in a new run dir (copy or link as `results/exp2/ROBUSTNESS.md` for the report).
  - `tests/test_robustness.py`: planted label_map-only effect is classified holds vs weakens; permutation null rejects it; deterministic.
  - Smoke check: `results/20261027-1510_exp2robust_qwen3.5-0.8b_smoke/` (plumbing only; tiny set, nearly all cells weakens).
  - **Remaining:** run on the real val/test Exp 2 runs; second model size (needs model runs); student sign-off on D-015.
