---
id: T-033
title: Prompt- and task-level prediction of generalization (H3)
status: in-progress
phase: 3
priority: P1
depends_on: [T-031]
owner: claude
estimate: M
tags: [analysis, exp2, prediction]
updated: 2026-10-26
---

# T-033 — Prompt- and task-level prediction of generalization (H3)

## Context
H3: does the **strength** of the representational shift predict accuracy on held-out queries, beyond what neutral geometry already predicts? This ticket also builds the shared prediction framework that T-043 reuses for the J-lens comparison, so the splits and metrics are identical.

## Tasks
- [ ] Shared framework (`src/icl_geometry/analysis/prediction.py`): given a feature table and an outcome, fit on train, select on validation, evaluate on test. Item-held-out CV. Metrics: held-out R² (logit margin), AUROC (incongruent correctness), Spearman (task level).
- [ ] **Task level:** across domain × cued attribute × m, does ΔSNR_cued predict incongruent error beyond neutral SNR (Exp 1 predictor)?
- [ ] **Prompt level:** predict margin/correctness from the per-prompt shift score; mixed-effects model with random intercepts for query item and demo set (statsmodels).
- [ ] Predictor families on identical splits:
  - neutral SNR;
  - in-context SNR;
  - shift score;
  - behavioral baselines (label bias, demo-order effects);
  - the **circular upper reference**: logit-lens/late-layer readout at the decision position, reported separately and labelled as such.
- [ ] Results table and figure; short write-up in `results/exp2/H3_REPORT.md`.

## Acceptance criteria
- [ ] The primary H3 metric (PLAN §6.9 item 2) is reported on the test split with a CI.
- [ ] The framework is reusable (T-043 imports it) and has tests on synthetic data.

## Outputs
- `src/icl_geometry/analysis/prediction.py`, `results/exp2/H3_REPORT.md`, `figures/exp2_h3_*`.

## Notes & pitfalls
- Don't let any predictor see test-split items during axis fitting or layer selection.
- Report the effect size (incremental R²/AUROC), not just significance.

## References
- PLAN §6.6, §6.8–6.9

## Work log
- 2026-10-26 (claude): `src/icl_geometry/analysis/prediction.py` (`evaluate`: fit on val, score on test, AUROC/R² with item-bootstrap CIs; `incremental`: paired bootstrap of the gain; `mixed_effects`: MixedLM with item intercept + demo-set variance component), tests `tests/test_prediction.py` (3 pass). `scripts/run_exp2_h3.py`: cued axes fit on val main prompts, layer selected by val AUROC, test AUROC of baseline (label bias + m), shift score, both, and the circular last-layer decision projection (reported separately), plus the incremental gain. Smoke-tested end to end. Not run on real data.
