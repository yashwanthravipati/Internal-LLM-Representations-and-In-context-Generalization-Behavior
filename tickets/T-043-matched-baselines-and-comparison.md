---
id: T-043
title: "Dimension-matched baselines & predictor comparison (H5)"
status: in-progress
phase: 4
priority: P1
depends_on: [T-042, T-033]
owner: claude
estimate: M
tags: [jlens, exp3, analysis, milestone]
updated: 2026-10-26
---

# T-043 — Dimension-matched baselines & predictor comparison (H5)

## Context
"J-space predicts better than full hidden states" means nothing unless lower-dimensional baselines are ruled out. Dimensionality reduction alone can change both SNR and predictive fits. This ticket delivers the H5 verdict and **milestone M4**.

## Tasks
- [ ] Baselines with the same dimensionality as each J-space representation:
  - **random subspaces** (many draws; compare with the Eq. 2 prediction);
  - **PCA subspace** (unsupervised, fit on training prompts);
  - **supervised probe direction(s)** (an upper reference, not a fair competitor);
  - **logit-lens space** (J = I);
  - optionally the tuned lens (Belrose et al. 2023).
- [ ] Use the T-033 framework with identical splits and metrics: held-out R², AUROC and Spearman, for Exp 1 tasks and Exp 2 prompts.
- [ ] Final comparison table and figure: J-space (a/b/c) vs full hidden vs matched baselines, by layer.
- [ ] Verdict on H5, with CIs; write `results/exp3/COMPARISON.md`.

## Acceptance criteria
- [ ] The primary H5 metric (PLAN §6.9 item 3) is reported on the test split with CIs against matched random subspaces.
- [ ] Table and figure produced by a script; status → `review`.

## Outputs
- `src/icl_geometry/analysis/exp3_compare.py`, `results/exp3/COMPARISON.md`, `figures/exp3_compare_*`.

## Notes & pitfalls
- Report the random-subspace **distribution** (e.g. a 95% band), not a single draw.
- If J-space ≈ random subspaces, that's a clean, reportable null.

## References
- PLAN §2 (H5), §6.6–6.7; Sorscher Eq. 2

## Work log
- 2026-10-26 (claude): `src/icl_geometry/geometry/subspaces.py` (random orthonormal, PCA and mean-difference bases; test shows Eq. 2 holds at m → ∞, M ≪ d, but **overstates the loss at finite m**, so the empirical random-subspace null is primary; documented in `docs/background/sorscher_theory.md`). `scripts/run_exp3_jspace.py`: per lens layer, representations full / J-space reconstruction / attribute-word J-lens span / logit-lens span / PCA-r / random-r (r = rank of the attribute span); held-out shift-score AUROC (axes fit on val) and test ΔSNR_cued; J-space variance explained and D. Smoke-tested on 0.8B (tiny subsample). Not run on real data.
