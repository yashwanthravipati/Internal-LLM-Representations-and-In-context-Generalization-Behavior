---
id: T-013
title: Zero-shot knowledge verification
status: todo
phase: 1
priority: P0
depends_on: [T-012, T-015]
owner: unassigned
estimate: M
tags: [tasks, behavior, controls]
updated: 2026-10-25
---

# T-013 — Zero-shot knowledge verification

## Context
If a model doesn't know that a duck is a bird, a failed in-context trial is a **knowledge** failure, not a **learning** failure. Filter items per model so that ICL errors reflect a failure to learn the grouping. The same explicit-question prompts double as the **positive control** for the J-lens rule readout (T-041): the J-lens paper found implicit properties enter the workspace when the question asks for them.

## Tasks
- [ ] For every item × attribute (Exp 2) and item × category (Exp 1), ask zero-shot forced-choice questions in ≥ 2 phrasings, with both option orders. Example: `Q: Is a duck a mammal or a bird?\nA:`, comparing log p(" mammal") vs log p(" bird"). Reuse the T-015 scoring harness.
- [ ] Pass rule (decide and log): correct in all phrasings and orders (strict), or above a margin threshold. Record the margin anyway.
- [ ] Per-model validity tables: `data/items/knowledge_<model>.parquet` (item, attribute, pass, margins).
- [ ] Report pass rates per domain/cell/category; flag cells that fall below the minimum counts.
- [ ] Save the explicit-question prompt set for reuse in T-041.

## Acceptance criteria
- [ ] Validity tables exist for all main models.
- [ ] A pass-rate report exists (`results/knowledge_check/REPORT.md`).
- [ ] The prompt generator (T-014) can filter by these tables.

## Outputs
- `src/icl_geometry/tasks/knowledge.py`, `data/items/knowledge_<model>.parquet`, the report, the explicit-question prompt set.

## Notes & pitfalls
- Don't filter the numbers domain on knowledge (it should pass trivially). If it doesn't, that's informative about the model.
- Low pass rates in a domain for the 0.8B model are expected. Decide per model whether the domain stays.

## References
- PLAN §4.7, §10; Gurnee et al. 2026 (implicit properties appear when asked)

## Work log
