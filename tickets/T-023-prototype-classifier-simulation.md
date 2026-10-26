---
id: T-023
title: Empirical prototype classifier simulation
status: in-progress
phase: 2
priority: P0
depends_on: [T-022]
owner: claude
estimate: S
tags: [geometry, theory, core]
updated: 2026-10-26
---

# T-023 — Empirical prototype classifier simulation

## Context
There are three numbers per task (PLAN §6.2–6.3): theory error, **empirical prototype-learner error on the same representations**, and LLM error. Theory vs prototype learner checks the math and the approximation; prototype learner vs LLM is the scientific question ("is ICL prototype-like?").

## Tasks
- [ ] `src/icl_geometry/geometry/prototype.py`: `prototype_error(Xa, Xb, m, n_draws, rng)`. Sample m training items per class, form prototypes (means), classify held-out items by nearest prototype (Euclidean), as in paper §L. Return the error for test examples of a and of b separately.
- [ ] Vectorize over pairs × layers × m (it will be called a lot).
- [ ] Validation figure (the analogue of paper Fig. 3A): theory vs simulated error across pairs and layers on LLM neutral reps (once T-021 exists; synthetic data before that).
- [ ] Note where theory and simulation diverge (low D, few samples, heavy-tailed manifolds).

## Acceptance criteria
- [ ] Function plus tests (the synthetic case matches T-022 tests; deterministic with a seed).
- [ ] Validation figure saved; deviations described in the Work log.

## Outputs
- `src/icl_geometry/geometry/prototype.py`, `tests/test_prototype.py`, `figures/theory_vs_prototype_*.png`.

## Notes & pitfalls
- Optionally add an exemplar (1-nearest-neighbor) learner and a linear SVM as alternative "learning rules". Comparing them against LLM error asks *which rule* ICL most resembles. This is cheap and informative.

## References
- Sorscher et al. 2022 Methods §L; PLAN §6.3

## Work log
- 2026-10-26 (claude): `src/icl_geometry/geometry/prototype.py`: `prototype_error(Xa, Xb, m, n_draws, rng, rule)` with disjoint train/test per draw (matches held-out queries), vectorized over draws; rules `prototype` (nearest mean) and `exemplar` (1-NN). Ties count 0.5. Tests (`tests/test_prototype.py`): separable/chance cases, seed determinism, exact equality with a naive per-draw loop for both rules. Synthetic validation figure: `figures/theory_vs_prototype_synthetic.png` (see T-022 log for numbers). Divergence: theory overestimates error for low-D (power-law) manifolds by ~0.05.
  - **Remaining:** the validation figure on LLM neutral representations needs T-021. Linear-SVM rule not added yet (optional). Status stays in-progress.
