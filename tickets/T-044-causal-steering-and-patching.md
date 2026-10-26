---
id: T-044
title: (Stretch) Causal steering & lens-coordinate patching
status: todo
phase: 4
priority: P2
depends_on: [T-041]
owner: unassigned
estimate: M
tags: [jlens, exp3, causal, stretch]
updated: 2026-10-25
---

# T-044 — (Stretch) Causal steering & lens-coordinate patching

## Context
Lenses are observational decoders: decodable ≠ used (course Project 10's theme; the neuroscience decoding-vs-perturbation debate). If adding or swapping the rule's concept vector **flips the model's grouping**, the J-lens features are causally involved, not just correlated. This is the analogue of optogenetic perturbation. **Start only after M2** (PLAN §8 scope rule).

## Tasks
- [ ] **Steering:** h ← h + α·v_word at the `query_item` / `decision` positions in the layer band where the rule readout peaks (T-041). Example: in a cue-A prompt, add v_"wild" (an attribute-B word); does the model start answering by attribute B? Measure the flip rate on incongruent queries vs α.
- [ ] **Matched control:** random directions with the same norm, and an unrelated word's concept vector.
- [ ] **Lens-coordinate patching** (paper method): swap the coordinates of v_s and v_t (e.g. "mammal" ↔ "wild"), leaving the orthogonal complement intact.
- [ ] **Ablation:** project out v_cued words; does accuracy drop more than for random directions?
- [ ] Dose-response plots and a short write-up in `results/exp3/CAUSAL.md`.

## Acceptance criteria
- [ ] Dose-response curves with controls; a clear statement of whether the effects exceed the controls.

## Outputs
- `src/icl_geometry/lens/interventions.py`, `results/exp3/CAUSAL.md`, `figures/exp3_causal_*`.

## Notes & pitfalls
- Large α breaks the model (label mass collapses). Track label mass alongside the flip rate.

## References
- `docs/background/jacobian_lens.md` (interventions); course Project 10 framing

## Work log
