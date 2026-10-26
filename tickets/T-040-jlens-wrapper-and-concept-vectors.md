---
id: T-040
title: J-lens wrapper, readouts & concept vectors
status: in-progress
phase: 4
priority: P1
depends_on: [T-004, T-020]
owner: claude
estimate: M
tags: [jlens, exp3]
updated: 2026-10-26
---

# T-040 — J-lens wrapper, readouts & concept vectors

## Context
Exp 3 needs the lens applied to **our cached activations**, not only through `jlens.apply` on raw prompts, plus per-concept directions for scoring and steering. Definitions are in `docs/background/jacobian_lens.md`.

## Tasks
- [ ] `src/icl_geometry/lens/jlens_wrapper.py`:
  - load the lens for a given model (path/revision from D-003);
  - `readout(h, layer)` → logits = W_U · norm(J_ℓ h), matching `jlens` **exactly**, including the final norm;
  - top-k decoding.
- [ ] `concept_vectors(tokens, layer)` → v_t = J_ℓᵀ W_U[t] for attribute words, synonyms and labels; `concept_scores(h, ...)` → ⟨v_t, h⟩ and cosine.
- [ ] A logit-lens readout (J = I) through the same interface, as a baseline.
- [ ] Verify that the layer indexing and hook point match T-020's cache convention (from T-004).
- [ ] Tests: our `readout` equals `jlens.apply` on the same prompt, layer and position within tolerance; concept vectors have the expected shape and ordering.
- [ ] Demo: readouts at `query_item` and `decision` for a few Exp 2 prompts across layers, saved as a slice-style table or figure.

## Acceptance criteria
- [ ] Tests pass (the match with `jlens.apply` is the key one).
- [ ] Demo readouts saved under `results/exp3/demo/`.

## Outputs
- `src/icl_geometry/lens/jlens_wrapper.py`, `tests/test_jlens_wrapper.py`, demo outputs.

## Notes & pitfalls
- Because of norm(·), ⟨v_t, h⟩ is a linear approximation of the lens logit. Use it for ranking within a position; compare across positions or layers with care (z-score within a position, or use ranks).
- Keep a minimal reimplementation that doesn't depend on `jlens` internals beyond loading J_ℓ. The package is unmaintained.

## References
- `docs/background/jacobian_lens.md`; PLAN §6.7

## Work log
- 2026-10-26 (claude): `src/icl_geometry/lens/jlens_wrapper.py`: `Lens.load` (model → lens file map from D-003), `Lens.logit_lens` (J = I baseline), `transport`, `readout` (= lm_head(final_norm(J h))), `top_tokens`, `concept_vectors` (v_t = J_lᵀ (g ⊙ W_U[t]), where g is the **effective final-norm gain** measured as norm(1); Qwen3.5's RMSNorm stores a zero-centred weight), `concept_scores` (dot + cosine), `state_index(l) = l + 1`. Only the lens *file* comes from jlens.
  - Tests `tests/test_jlens_wrapper.py`: our readout on cached states equals `jlens.apply` (atol 1e-3) at layers 3/12/20 and two positions; concept-vector dot products rank tokens **exactly** like the lens logits. Both pass.
  - Remaining: demo readouts for Exp 2 prompts under `results/exp3/demo/` (after T-030).
