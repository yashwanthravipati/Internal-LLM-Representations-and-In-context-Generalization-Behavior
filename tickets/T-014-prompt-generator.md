---
id: T-014
title: Prompt generator (all conditions, counterbalanced, deterministic)
status: review
phase: 1
priority: P0
depends_on: [T-010, T-012]
owner: claude
estimate: M
tags: [tasks, prompts]
updated: 2026-10-26
---

# T-014 — Prompt generator (all conditions, counterbalanced, deterministic)

## Context
All experiments run on prompt sets produced here. The generator must cover every condition in PLAN §4.6 and be deterministic, so results are reproducible. It also fixes the held-out item splits (D-009).

## Tasks
- [ ] Default template (may change in T-016):
  ```
  dog -> foo
  chicken -> bar
  wolf -> foo
  eagle -> bar
  duck ->
  ```
  One demo per line; the last line is the query; the label is predicted at the final position.
- [ ] **Exp 1 tasks:** a category pair (a, b), m ∈ {1, 2, 4, 8} demos per category, demos shuffled, query drawn from the held-out query pool.
- [ ] **Exp 2 tasks:** domain plus cued attribute ∈ {A, B}. Demos are **balanced on the uncued attribute**, so it carries no information. Queries are congruent or incongruent, with the congruency flag computed and stored.
- [ ] **Controls:**
  - random-label context (same items and format, labels shuffled so no rule fits);
  - explicit-rule variant (the rule stated in words before the demos);
  - ambiguous demos (A and B perfectly confounded in the demos; for Exp 4/T-050).
- [ ] **Counterbalancing:** both label→group mappings; ≥ 3 demo orders; ≥ 2 label sets.
- [ ] **Splits (D-009):** a fixed, seeded split of items per category/cell into demo pool vs query pool, plus a train/validation/test assignment for analyses. Store the split in each record.
- [ ] Optional knowledge filtering from T-013 tables, applied per model.
- [ ] Record fields (PLAN §7.3): `prompt_id, experiment, domain, category_pair | cued_attr/uncued_attr, m, label_set, label_map, demos[{item,label}], query_item, query_attrs, congruent, condition, correct_label, incorrect_label, template_id, seed, split, text`.
- [ ] CLI: `uv run python scripts/make_prompts.py --config configs/prompts/<name>.yaml`, writing JSONL.
- [ ] Unit tests: demos balanced on the uncued attribute; query never among the demos; label mapping and correct-label logic right; congruency logic right; identical output for the same config and seed.

## Acceptance criteria
- [ ] Generator and tests pass.
- [ ] Sample prompt sets (≥ 200 prompts each) exist for Exp 1 and Exp 2, with all conditions represented.
- [ ] D-009 status → accepted, with the split procedure described.
- [ ] Example prompts shown in the ticket Work log.

## Outputs
- `src/icl_geometry/tasks/prompts.py`, `scripts/make_prompts.py`, `configs/prompts/*.yaml`, `tests/test_prompts.py`.

## Notes & pitfalls
- Recency and majority-label biases are strong in small models. Balance the label counts and vary which label appears last.
- Keep query-item positions easy to recover. Either store character offsets now, or have T-020 recompute token positions robustly.

## References
- PLAN §4.6, §6.8, §7.3; Zhao et al. 2021; Min et al. 2022

## Work log
- 2026-10-26 (claude): `src/icl_geometry/tasks/prompts.py` (`PromptConfig`, `assign_splits`, `generate_exp1`, `generate_exp2`, `generate`), `src/icl_geometry/tasks/templates.py`, `scripts/make_prompts.py`, `tests/test_prompts.py` (9 tests: query ∉ demos and span correct; Exp 2 demos balanced on the uncued attribute and labels follow the cued one; congruency logic; random-label breaks both rules; ambiguous confounds them; label maps counterbalanced; deterministic; splits stable/disjoint; val/test item sets disjoint). All pass.
  - Design notes: **Exp 2 requires even m** (balance on the uncued attribute), so the Exp 2 grid is m ∈ {2, 4, 8}; m = 1 per label is by construction the ambiguous condition. Congruency convention: value index 0 of each attribute aligned to the same label (`mammal`/`domestic` → label 0), as in PLAN §4.3. Queries per demo set: one per crossed cell (2 congruent + 2 incongruent). Exp 1 queries: `n_queries_per_class` per category. Records carry `query_span` (char offsets), `demo_set` and `order` for mixed-effects models.
  - Pilot sets (val split only): `configs/prompts/exp2_pilot_{arrow,io}.yaml` (3168 / 1056 prompts), `exp1_pilot_{arrow,io}.yaml` (2112 each; 12 categories, 66 pairs). Countries has no m = 8 prompts on the val split (Asia-landlocked too small).
  - Example (animals, cue = class, m = 2, main):
    ```
    hawk -> bar
    parakeet -> bar
    cheetah -> foo
    sheep -> foo
    hamster ->
    ```
    Explicit-rule adds `Rule: label by class. mammal -> foo, bird -> bar.` before the demos.
  - D-009 (splits) described below; status stays `proposed` until the student confirms the 40/60 val/test split.
