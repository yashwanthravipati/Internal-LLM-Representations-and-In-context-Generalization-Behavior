---
id: T-042
title: J-space projection & J-space geometry
status: in-progress
phase: 4
priority: P1
depends_on: [T-040, T-031]
owner: claude
estimate: L
tags: [jlens, exp3, geometry]
updated: 2026-10-26
---

# T-042 — J-space projection & J-space geometry

## Context
H5 asks whether geometry computed in the J-space (the low-variance, verbalizable subspace) predicts generalization better than geometry of the full hidden state. This ticket builds the J-space representations and recomputes the geometry metrics on them. T-043 makes the comparison fair.

## Tasks
- [ ] Sparse non-negative decomposition onto J-lens vectors at each layer (k ≤ 25, gradient pursuit). **Check `jlens` for an implementation first** (T-004 notes); otherwise implement it, with a test against a brute-force small case.
- [ ] Three J-space representations:
  - (a) the J-space component of h (reconstruction from the sparse code);
  - (b) the sparse codes themselves (vocabulary-indexed features);
  - (c) the projection onto the span of the attribute-word concept vectors (a small, hypothesis-driven subspace).
- [ ] Recompute on each representation: the Sorscher SNR (Exp 1 tasks and Exp 2 cued partitions), ΔSNR_cued, and per-prompt shift scores (T-031 functions).
- [ ] Track the variance explained by the J-space component per layer (the paper reports ≲ 10%) and its effective dimension (D).
- [ ] Cache the features for the Exp 2 prompts (and the Exp 1 neutral reps).

## Acceptance criteria
- [ ] The decomposition is implemented or borrowed, and tested.
- [ ] J-space features cached; the geometry metrics computed; sanity plots of variance explained and D vs layer.

## Outputs
- `src/icl_geometry/lens/jspace.py`, `tests/test_jspace.py`, caches, `results/exp3/JSPACE_SANITY.md`.

## Notes & pitfalls
- **Key tension (PLAN §6.7):** projecting to fewer dimensions lowers D, which *raises* the D⁻¹/m noise term in Eq. 1. J-space SNR can be lower even if the J-space is "the right space". Interpret only against dimension-matched baselines (T-043).
- Gradient pursuit over a vocabulary of ~150k vectors per position is expensive. Restrict the dictionary (e.g. the top-N tokens by lens score, or a curated concept vocabulary) if needed, and document the restriction.

## References
- `docs/background/jacobian_lens.md` (J-space); Sorscher Eq. 1–2; PLAN §6.7

## Work log
- 2026-10-26 (claude): `jlens` has no decomposition, so implemented `src/icl_geometry/lens/jspace.py`: `concept_vocabulary` (space-prefixed lowercase words ≥ 3 letters + required ids), `build_dictionary` (unit-normalized concept vectors per layer), batched non-negative **gradient pursuit** (k ≤ 25, exact line search, clipping), `SparseCode` (indices, coefs, recon, dense codes), `variance_explained`, `span_projection` (attribute-word subspace). Tests `tests/test_jspace.py`: exact recovery of planted 5-sparse codes (VE > 0.99); vs exhaustive NNLS on a tiny dictionary, never better than the optimum and within 10% in ≥ 70% of cases; span rank. All pass.
  - Remaining: run on cached states, geometry on the three J-space representations, sanity report.
