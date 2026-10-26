---
id: T-016
title: Pilot behavioral sweep & difficulty tuning
status: todo
phase: 1
priority: P0
depends_on: [T-013, T-015]
owner: unassigned
estimate: M
tags: [behavior, pilot, milestone]
updated: 2026-10-25
---

# T-016 — Pilot behavioral sweep & difficulty tuning

## Context
Geometry can only predict behavior if behavior **varies**. If every task is at ceiling, or at chance, there is nothing to predict. This pilot checks dynamic range and fixes the final task parameters. Completing it is **milestone M1**.

## Tasks
- [ ] Run small Exp 1 and Exp 2 prompt sets on the dev and main candidate models.
- [ ] Plots:
  - accuracy vs m per task (learning curves);
  - congruent vs incongruent accuracy per Exp 2 domain and cued attribute;
  - label mass;
  - random-label control accuracy (should be at chance on rule-dependent queries);
  - explicit-rule control (should be higher).
- [ ] Check the dynamic range. Exp 1 errors should spread across tasks. Exp 2 incongruent accuracy should sit clearly above chance and below ceiling for at least some conditions. If not, adjust: harder or easier pairs, the m grid, label sets, the template, or the model size.
- [ ] Choose the final template, label sets, m grid, domains and main model(s); log them as **D-010**.
- [ ] Estimate the final sweep size and compute (prompts × models × positions), and add it to `docs/COMPUTE.md`.

## Acceptance criteria
- [ ] `results/pilot/REPORT.md` with figures and a short interpretation.
- [ ] D-010 logged; D-003 (model choice) accepted.
- [ ] Status → `review`: the student confirms the final task parameters before the big sweeps.

## Outputs
- Pilot run directories, `results/pilot/REPORT.md`, DECISIONS D-010.

## Notes & pitfalls
- Don't over-tune on the pilot. Pilot prompts should use items from the **validation** split only, never the test split.
- An attribute where the model can't learn the regrouping at all is still a finding, but keep at least two domains where it can.

## References
- PLAN §4.6, §8 (M1), §10

## Work log
- 2026-10-26 (claude): Infrastructure only, per the student (experiments to be run later): configs `configs/prompts/exp{1,2}_pilot_{arrow,io}.yaml`, `configs/runs/pilot_*`, analysis `scripts/analyze_pilot.py`. A partial run (0.8B only, all 4 sets) was started by mistake and stopped. Its outputs in `results/20261026-01{25,36,43,50}_pilot_qwen3.5-0.8b_*` can serve as smoke tests of the harness but are **not** the official pilot. Nothing was run on 2B-Base.
