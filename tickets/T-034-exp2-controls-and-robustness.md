---
id: T-034
title: Controls & robustness for Experiment 2
status: todo
phase: 3
priority: P1
depends_on: [T-032, T-033]
owner: unassigned
estimate: M
tags: [analysis, exp2, controls]
updated: 2026-10-25
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
