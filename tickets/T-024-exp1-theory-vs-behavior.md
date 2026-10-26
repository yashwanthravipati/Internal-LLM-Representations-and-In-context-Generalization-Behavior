---
id: T-024
title: "Experiment 1 analysis: theory vs LLM few-shot behavior (MVP)"
status: todo
phase: 2
priority: P0
depends_on: [T-016, T-021, T-023]
owner: unassigned
estimate: L
tags: [analysis, exp1, milestone, core]
updated: 2026-10-25
---

# T-024 — Experiment 1 analysis: theory vs LLM few-shot behavior (MVP)

## Context
This is **the core Project 5 deliverable and milestone M2.** Does Sorscher-theory error computed from neutral item representations predict LLM in-context error across category pairs and shot counts (H1)? Lock the primary metrics (PLAN §6.9, D-005) **before** running the final analysis.

## Tasks
- [ ] Set D-005 to accepted (lock the primary metrics) before looking at test-split results.
- [ ] Final Exp 1 behavioral sweep: all category pairs × m ∈ {1, 2, 4, 8} × counterbalancing, on the main model(s) (T-015 harness).
- [ ] For each task (pair, direction a→b, m) and each layer, compute: theory error (Eq. 1), prototype-simulation error (T-023), and LLM error (raw and calibrated).
- [ ] **Layer selection on validation** (held-out categories/items); headline numbers on **test**.
- [ ] Figures:
  1. predicted vs LLM error scatter at the selected layer;
  2. Spearman ρ by layer (full profile, normalized depth);
  3. learning curves vs m, theory vs LLM;
  4. term decomposition: how much each geometric term (signal, bias, D, overlaps) explains;
  5. **asymmetry test:** theory-predicted ε_a − ε_b vs LLM asymmetry, controlling for label bias.
- [ ] Baselines: word frequency; category size; signal-only (centroid distance) predictor; layer-0 (embedding) geometry. Does the full Eq. 1 beat the signal-only predictor?
- [ ] Optional: compare LLM error with exemplar/SVM learners (T-023 note) to ask which learning rule ICL resembles.
- [ ] Write `results/exp1/REPORT.md`: Spearman ρ with bootstrap CIs over categories, the shuffled baseline, and an interpretation in plain language (including if H1 fails).

## Acceptance criteria
- [ ] The primary metric (PLAN §6.9 item 1) is computed on the held-out split and reported with a CI and the shuffled baseline.
- [ ] All five figures produced by a script (not just a notebook).
- [ ] Report written; status → `review` for the student to read the interpretation.

## Outputs
- `src/icl_geometry/analysis/exp1.py`, `scripts/run_exp1_analysis.py`, `results/exp1/REPORT.md`, `figures/exp1_*`.

## Notes & pitfalls
- Tasks share categories, so they aren't independent. Bootstrap over **categories**, not tasks.
- If the theory predicts the prototype learner well but the LLM poorly, that's a real result: ICL isn't prototype-like at these layers. Discuss why.
- Start the report draft (T-061) as soon as this lands.

## References
- PLAN §2 (H1), §4.2, §6.2–6.3, §6.8–6.9

## Work log
