---
id: T-015
title: Behavioral scoring harness
status: in-progress
phase: 1
priority: P0
depends_on: [T-014, T-003]
owner: claude
estimate: M
tags: [behavior, models]
updated: 2026-10-26
---

# T-015 — Behavioral scoring harness

## Context
This turns prompts into behavioral measurements: the dependent variables every analysis predicts (PLAN §6.1).

## Tasks
- [ ] Batched forward pass that returns log-probabilities of the two candidate label tokens at the decision position. **Get padding right:** left padding with correct attention masks and position IDs, or right padding with per-row last-token indexing.
- [ ] Per-prompt record: `logp_correct, logp_incorrect, margin = logp_correct − logp_incorrect, correct = margin > 0, label_mass = p(label1) + p(label2)`.
- [ ] Contextual calibration option (Zhao et al. 2021): estimate the label bias with content-free queries (e.g. "N/A") for the same demos, and report both calibrated and raw margins.
- [ ] Save `behavior.parquet` keyed by (prompt_id, model) into the run directory (T-005 conventions).
- [ ] CLI: `uv run python scripts/run_behavior.py --config configs/runs/<name>.yaml`.
- [ ] Tests: **padding invariance** (a prompt scored alone vs inside a batch gives the same logits within tolerance); label tokens are resolved correctly.

## Acceptance criteria
- [ ] The harness scores a 1k-prompt set on the dev model.
- [ ] The padding-invariance test passes.
- [ ] Label-mass sanity is reported (median label mass should be high; if not, the model isn't following the format).

## Outputs
- `src/icl_geometry/behavior/scoring.py`, `scripts/run_behavior.py`, `tests/test_scoring.py`.

## Notes & pitfalls
- Labels are scored with their leading space (" foo"), matching how they appear in the demos.
- With post-trained Qwen checkpoints, use raw completion (no chat template) unless DECISIONS says otherwise.

## References
- PLAN §6.1; Zhao et al. 2021

## Work log
- 2026-10-26 (claude): `src/icl_geometry/behavior/scoring.py`: right-padded batched forward (sorted by length) giving label log-probs at the last real token, margin, correct, label mass; contextual calibration with the content-free query "N/A" under the same demos (and the same rule header for explicit-rule prompts): `cf_margin`, `margin_calibrated`, `correct_calibrated`; optional residual states at `query_item`/`decision` (shared with T-020). `scripts/run_behavior.py` (RunConfig YAML → `results/<run_id>/behavior.parquet`, `summary.json`). Tests `tests/test_scoring.py`: padding invariance (batch vs single, margins within 1e-3, states within fp16 tolerance), log-probs equal a direct forward pass, content-free text. All pass.
  - The 1k-prompt acceptance run happens in the pilot (T-016), currently running.
