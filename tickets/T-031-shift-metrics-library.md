---
id: T-031
title: Shift metrics library (ΔSNR, selection index, RSA, CCGP, per-prompt shift)
status: in-progress
phase: 3
priority: P1
depends_on: [T-022, T-030]
owner: claude
estimate: M
tags: [geometry, exp2]
updated: 2026-10-26
---

# T-031 — Shift metrics library (ΔSNR, selection index, RSA, CCGP, per-prompt shift)

## Context
This quantifies "how much the representation reorganized toward the demonstrated grouping" in several complementary ways (PLAN §6.4). Each metric is tied to a neuroscience tradition: Sorscher SNR, probing with control tasks, RSA (Kriegeskorte), CCGP (Bernardi).

## Tasks
Implement in `src/icl_geometry/geometry/`:
- [ ] **ΔSNR_cued**: the Sorscher SNR for the cued partition, computed on in-context query reps, minus the same quantity on (a) neutral reps and (b) random-label-context reps. In-context manifold = query reps across prompts with the same (domain, cued attribute, m, condition), grouped by the query's cued value.
- [ ] **Selection index**: cross-validated linear-probe accuracy (or d′) for the cued attribute minus the uncued attribute, per layer and position. Include **control tasks** (Hewitt & Liang 2019) to calibrate probe selectivity.
- [ ] **RSA**: item × item dissimilarity (correlation distance) per context; partial Spearman correlation with the attribute-A vs attribute-B model RDMs.
- [ ] **CCGP**: train a decoder for the cued attribute on items with one value of the uncued attribute; test on the other value.
- [ ] **Per-prompt shift score**: projection of (in-context − neutral) query rep onto the cued-attribute axis. The axis is estimated on **training** prompts only (e.g. a difference of class means); normalized.
- [ ] Unit tests with synthetic data where a known shift is planted (and a null case with none).

## Acceptance criteria
- [ ] Library plus tests pass (planted shift detected; null gives ~0).
- [ ] Smoke run on one model and domain, with outputs saved.

## Outputs
- `src/icl_geometry/geometry/{shift.py,probes.py,rsa.py,ccgp.py}`, `tests/test_shift.py`.

## Notes & pitfalls
- Probe accuracy saturates in high dimensions with few samples. Use regularization chosen by nested CV, and report the control-task selectivity.
- The context changes residual streams for many reasons (more tokens, label tokens). That's why the **random-label control** is the primary comparison, not neutral reps.

## References
- PLAN §6.4; Hewitt & Liang 2019; Kriegeskorte et al. 2008; Bernardi et al. 2020

## Work log
- 2026-10-26 (claude): `src/icl_geometry/geometry/shift.py`: `snr_partition`/`delta_snr` (Sorscher SNR of the cued partition, both directions, full Eq. 1), `probe_accuracy`/`selection_index` with item-grouped CV and a Hewitt–Liang control task (`control_labels`: a random label per item, constant across its prompts), `rsa_partial` (partial Spearman of the item RDM with each attribute's model RDM), `ccgp`, `shift_axis`/`shift_scores` (signed projection of in-context − neutral onto the cued axis, in units of half the class separation). Tests `tests/test_shift.py` (planted vs null shift, selection + control, RSA, CCGP factorized vs XOR, shift sign/scale, control labels): 7 pass.
  - **Design change:** decoders default to **nearest class mean (prototype)**, not logistic regression. In the synthetic test, a logistic probe on held-out items reached only 0.85 accuracy where the class means separate perfectly (item-specific noise × few items × high d), while the prototype decoder reached 1.0. Logistic is still available (`decoder="logistic"`). Files are a single `shift.py` rather than four modules.
  - Smoke run on real data waits for T-030 activations.
