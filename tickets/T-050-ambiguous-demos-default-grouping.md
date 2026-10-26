---
id: T-050
title: (Stretch) Ambiguous demonstrations & default grouping
status: todo
phase: 5
priority: P2
depends_on: [T-016, T-031]
owner: unassigned
estimate: M
tags: [exp4, stretch, behavior, geometry]
updated: 2026-10-25
---

# T-050 — (Stretch) Ambiguous demonstrations & default grouping

## Context
When the demonstrations fit **both** rules (A and B perfectly confounded in the demos), which grouping does the model adopt? This parallels default dimensions in human category learning (e.g. children's shape bias). Prediction: the model defaults to the attribute with higher **neutral** SNR (the more "geometrically salient" grouping). **Start only after M2.**

## Tasks
- [ ] Ambiguous prompt sets (T-014 condition): demos in which A and B agree; probed with incongruent queries that reveal the adopted rule.
- [ ] Measure the preference for A vs B per domain, model and m.
- [ ] Test whether neutral SNR(A) − SNR(B) predicts the preference across domains (and across item subsets within a domain).
- [ ] Does in-context geometry (T-031 metrics) show which rule was adopted before the answer (at `query_item`, or in middle layers at `decision`)?
- [ ] Write `results/exp4/AMBIGUOUS.md`.

## Acceptance criteria
- [ ] Preference measured with CIs; the geometry-prediction test reported (including nulls).

## Outputs
- `results/exp4/AMBIGUOUS.md`, `figures/exp4_*`.

## References
- PLAN §4.5; Landau, Smith & Jones 1988 (shape bias); `docs/background/neuro_connections.md`

## Work log
