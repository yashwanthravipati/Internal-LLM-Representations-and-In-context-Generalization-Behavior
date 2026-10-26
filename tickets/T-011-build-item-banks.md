---
id: T-011
title: Build item banks
status: review
phase: 1
priority: P0
depends_on: [T-010]
owner: claude
estimate: L
tags: [tasks, data]
updated: 2026-10-26
---

# T-011 — Build item banks

## Context
Fill the schemas from T-010 with real items. The geometry estimates (participation ratio, radii) need enough samples per manifold, so the counts matter (PLAN §4.7, §10).

## Tasks
- [ ] **Exp 1:** pull category memberships from THINGS (check the license and download terms). Deduplicate; drop items in several categories (or assign a primary category and record it); drop proper nouns and abstract items.
- [ ] **Exp 2:** fill the crossed domains from feature norms (McRae 2005; CSLB; Binder 2016; THINGSplus ratings) or curated lists. Generate the numbers domain programmatically (e.g. 1–99, with or without two-digit-only constraints for tokenization).
- [ ] Record provenance (`source`) and `confidence` (high/medium/low) for every item. Exclude low-confidence items from the main analyses.
- [ ] Validation script `scripts/validate_items.py`: counts per category/cell, duplicates across files, balance, missing fields. Writes `results/item_bank_report.md`.
- [ ] Optional: cross-check the attribute labels with a strong LLM *or* a second human rater. **Flag** disagreements for manual review; don't auto-accept them.

## Acceptance criteria
- [ ] `data/items/*.csv` meets the T-010 count targets **before** model-specific filtering, with ~30% headroom (filters will remove items).
- [ ] The validation script passes; the report is generated.
- [ ] Status → `review` for the student to spot-check a random sample of 50 items.

## Outputs
- `data/items/categories.csv`, `data/items/<domain>.csv`, `scripts/validate_items.py`, `results/item_bank_report.md`.

## Notes & pitfalls
- Over-collect: T-012 (tokenization) and T-013 (zero-shot knowledge) will drop items, especially for the 0.8B model.
- Keep spelling consistent with how items appear in prompts (lowercase singular nouns by default).

## References
- PLAN §4.7; T-010

## Work log
- 2026-10-26 (claude): Built `data/items/categories.csv` (970 items, 31 categories), `animals.csv` (96), `countries.csv` (84), `words.csv` (78), `numbers.csv` (89, programmatic) from curated lists in `scripts/build_item_banks.py` (source `curated-v1`, confidence + notes per item; polysemous items flagged; items listed under two categories are kept once and marked `low`). `scripts/validate_items.py` passes hard checks and writes `results/item_bank_report.md`.
  - **Below target:** 9 Exp 1 categories have < 26 high-confidence items (lowest: footwear 14, herb 16, metal 18, toy 19, reptile 20). Exp 2 cells below 20: animals bird×domestic (6!), mammal×domestic (16); countries Asia×landlocked (8), Europe×landlocked (12); all words cells (17–19). The Asia-landlocked and domestic-bird cells are near the real-world maximum, so more curation can't fix them much.
  - Not done: THINGS cross-check / second rater. Status `review`: the student should spot-check ~50 random items (e.g. `uv run python -c "import pandas as pd; print(pd.read_csv('data/items/categories.csv').sample(50, random_state=0))"`).
