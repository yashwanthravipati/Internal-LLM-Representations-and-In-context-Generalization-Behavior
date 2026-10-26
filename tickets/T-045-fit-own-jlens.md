---
id: T-045
title: (Contingency) Fit our own Jacobian lens
status: todo
phase: 4
priority: P2
depends_on: [T-004]
owner: unassigned
estimate: M
tags: [jlens, contingency]
updated: 2026-10-25
---

# T-045 — (Contingency) Fit our own Jacobian lens

## Context
Do this **only if** T-004 finds that the pre-fitted lenses don't fit our needs: wrong checkpoint (e.g. we need the base model), J++ differs from the J-lens in a way that matters for our claims, or the files are broken. Otherwise close it as `wontfix`.

## Tasks
- [ ] Assemble a fitting corpus of ~100–1000 sequences of 128 tokens from a pretraining-like source (e.g. WikiText-103, as used for the pre-fitted lenses).
- [ ] `jlens.fit(model, prompts=..., checkpoint_path=...)` on the needed checkpoint (start with the smallest). Parallelize across disjoint slices with `JacobianLens.merge()` if useful.
- [ ] Validate with the Anthropic evaluation prompt sets in the jlens repo's `data/` folder (e.g. Recall@10, comparable to the HF card), and with the T-004 sanity readout.
- [ ] Save to `data/lenses/<model>/` (gitignored), with provenance (corpus, n_prompts, commit, date) in a JSON sidecar.

## Acceptance criteria
- [ ] A fitted lens passes the sanity checks; provenance recorded; D-003 updated to point at it.

## Outputs
- `scripts/fit_jlens.py`, `data/lenses/<model>/…`, a DECISIONS update.

## Notes & pitfalls
- Fitting is dominated by backward passes and is slow. Use the T-004 timing estimate to budget compute.

## References
- `docs/background/jacobian_lens.md`; github.com/anthropics/jacobian-lens

## Work log
