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
- 2026-10-27 (claude): `scripts/validate_theory_neutral.py`: theory (Eq. 1, full and dominant) vs Monte Carlo prototype and exemplar learners on cached neutral reps, 300 random ordered category pairs × 5 depths × m ∈ {1,2,4,8}. Reads the neutral cache only. Figure: `make_figures.py` → `theory_vs_prototype_neutral`.
  - Dev-model check (0.8B, `results/20261027-1509_t023_qwen3.5-0.8b_dev/`; a math check, not a result): Spearman ρ(theory, simulation) 0.85–0.99 per layer × m; mean |theory − simulation| 0.010–0.065, overall 0.028. At the embeddings (state 0) the theory *under*estimates error (mean bias −0.01 to −0.04); at later layers it *over*estimates at m = 1–2 (+0.01 to +0.04), and at m ≥ 4 the gap is ≤ 0.015. |error| falls with m (ρ = −0.40) and barely depends on D (ρ = 0.12) or P (ρ = −0.13). Prototype and exemplar errors agree closely at m = 1 (identical by construction) and diverge with m (ρ 0.77–0.82 at m = 8).
  - **Remaining:** rerun on 2B-Base neutral reps (needs T-021 on the main model) and point the manifest at it. Linear-SVM rule still not added (optional).
