---
id: T-022
title: Implement Sorscher geometry estimators & SNR theory
status: review
phase: 2
priority: P0
depends_on: [T-002]
owner: claude
estimate: M
tags: [geometry, theory, core]
updated: 2026-10-26
---

# T-022 — Implement Sorscher geometry estimators & SNR theory

## Context
This is the core of Project 5: implement the theory exactly, and validate it, before trusting any LLM comparison. Equations and estimation details are in `docs/background/sorscher_theory.md` (checked against the paper's full text). The ticket needs only the environment, so it can run in parallel with Phase 1.

## Tasks
- [ ] `src/icl_geometry/geometry/sorscher.py`:
  - `manifold_stats(X)` → centroid x₀, radii Rᵢ, axes uᵢ (non-zero only), mean squared radius R², participation ratio D. Use the **Gram-matrix trick** for P ≪ N.
  - `snr(stats_a, stats_b, m)` → SNR_a per Eq. 1, **plus each term** (signal, bias, dimension, overlap_a, overlap_b) for decomposition plots; `error = scipy.stats.norm.sf(snr)`.
  - Both directions (a vs b and b vs a); vectorized over m; the m → ∞ limit.
  - `random_projection_snr(snr_inf, D_inf, M)` per Eq. 2 (needed for the T-043 baselines).
- [ ] **Normalization:** port the conventions (1/N, 1/P factors in R² and Rᵢ) from the official code at github.com/bsorsch/geometry-fewshot-learning. Document each one in docstrings with equation numbers.
- [ ] Numerical cross-check against the official implementation on a shared example (port their function or run it side by side).
- [ ] Unit tests on synthetic ellipsoidal manifolds where the error is known by Monte Carlo (cf. paper Fig. 2E–H). Vary signal, bias (unequal radii), D and signal–noise overlap; check theory vs simulation within tolerance across m ∈ {1, 2, 5, ∞}.

## Acceptance criteria
- [ ] Tests pass. Theory is within a stated tolerance of Monte Carlo on synthetic manifolds in the regime where the paper claims accuracy (high D).
- [ ] Agreement with the official code to ~1e-6 on a shared example, or the differences are explained in the Work log.
- [ ] Docstrings cite the equations, and `docs/background/sorscher_theory.md` is updated with the resolved normalization.

## Outputs
- `src/icl_geometry/geometry/sorscher.py`, `tests/test_sorscher.py`.

## Notes & pitfalls
- With P ~ 20–60 samples, there are at most P − 1 non-zero eigenvalues. D is capped and biased low. Expose P so the analyses can report it.
- Compute in float64.
- The theory's "dominant terms" approximation may be poor at low D. Monte Carlo (T-023) is the ground truth for what the prototype learner does.

## References
- `docs/background/sorscher_theory.md`; Sorscher et al. 2022 Eq. 1, Eq. 2, Methods §K–L, SI §3C

## Work log
- 2026-10-26 (claude): Implemented `src/icl_geometry/geometry/sorscher.py`: `manifold_stats` (thin SVD of the centered P × N matrix = Gram trick cost; keeps the ≤ P − 1 non-zero radii; exposes P), `pair_terms`, `snr`/`snr_from_terms` (vectorized over m incl. `np.inf`; returns SNR, error = `norm.sf`, and each term: signal, bias, dim, overlap_a, overlap_b, ss), `pairwise_snr` (both directions), `random_projection_snr` and `projected_dimension` (Eq. 2). Float64 throughout.
  - Normalization ported from the official repo (commit fee5d47): signal normalized by tr(C_a) = ΣR²/P. Cross-check vs a verbatim numpy port of their `geometry()`: agreement to rtol 1e-8 (test). The official denominator includes a subspace–subspace term ss/m that the short-form Eq. 1 in our notes omitted; it's included. `full=True` adds the SI 1/m² terms.
  - Monte Carlo (via T-023's `prototype_error`): unit tests on isotropic, unequal-radii and signal–noise-overlap manifolds pass at |Δ| < 0.03 for m ∈ {1, 2, 5}; m → ∞ limit tested. Synthetic sweep `scripts/validate_theory_synthetic.py` → `results/20261026-0111_t022_synthetic_theory-vs-mc/` (summary.csv, parquet, figure): high-D MAE 0.005 (m = 1) → 0.001 (m = 10), r = 0.999; low-D (D ≈ 2.5) MAE ≈ 0.05, r ≈ 0.99, theory overestimates error. Docs updated (`docs/background/sorscher_theory.md`, normalization section).
  - Recommendation: use `full=True` as the primary theory prediction (lower error at small m); report the dominant-terms version as a sensitivity check. Logged as D-013 (proposed).
- 2026-10-26 (claude): Work complete and verified; status held at `review` only because a dependency (T-002 git init, or T-011 item spot-check) is still open. Flip to `done` once that closes.
