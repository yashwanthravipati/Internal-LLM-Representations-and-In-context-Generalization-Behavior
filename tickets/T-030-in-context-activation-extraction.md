---
id: T-030
title: In-context activation extraction (Exp 2)
status: in-progress
phase: 3
priority: P1
depends_on: [T-020, T-016]
owner: claude
estimate: M
tags: [activations, exp2]
updated: 2026-10-26
---

# T-030 — In-context activation extraction (Exp 2)

## Context
Exp 2 asks whether the representations of the **same items** change when the demonstrations cue a different grouping. That needs query-item and decision-position activations inside full prompts, linked one-to-one with behavior, plus control-condition activations.

## Tasks
- [ ] Generate the final Exp 2 prompt sets (T-014, parameters from D-010): main conditions (cue A, cue B; congruent and incongruent queries; all m; counterbalancing), the **random-label control**, and the **explicit-rule control**.
- [ ] Score behavior for every prompt (T-015).
- [ ] Extract activations at `query_item` and `decision`, all layers (or the D-010 layer subset), on the main model(s) (T-020).
- [ ] Make sure neutral reps (T-021) exist for every query item, for paired in-context vs neutral comparisons.
- [ ] Data-integrity check: every activation row ↔ prompt_id ↔ behavior row; counts per condition cell match the design.

## Acceptance criteria
- [ ] Caches plus indexes written; behavior parquet for the same prompts.
- [ ] The integrity-check script passes; a per-cell count table is in the Work log.

## Outputs
- Run directories under `results/`, caches under `data/cache/`, `scripts/check_integrity.py`.

## Notes & pitfalls
- Keep test-split prompts separate. Only validation prompts are used for any layer or axis selection downstream.

## References
- PLAN §4.3, §5, §7.3

## Work log
- 2026-10-26 (claude): Infrastructure ready, not yet run on the main model: final (provisional until D-010) prompt configs `configs/prompts/exp2_main_{val,test}.yaml` (8448 / 9216 prompts) and run configs `configs/runs/exp2_main_{val,test}_2b-base.yaml` (cache both positions); integrity checker `scripts/check_integrity.py` (prompt-set hash, prompt↔behavior↔activation one-to-one and order, finite values, per-cell counts). Smoke-tested on 0.8B with a tiny set (`results/20261026-0209_smoke_*`, `-0210_smoke_*`): integrity OK.
