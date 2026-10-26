---
id: T-010
title: Task & domain design (category pairs + crossed 2×2 domains)
status: review
phase: 1
priority: P0
depends_on: []
owner: claude
estimate: M
tags: [tasks, design, science]
updated: 2026-10-26
---

# T-010 — Task & domain design (category pairs + crossed 2×2 domains)

## Context
There are two kinds of tasks (PLAN §4.2–4.3):
- **Exp 1, category-pair tasks:** many pairs of categories (animal vs vehicle, fruit vs vegetable, bird vs insect …). They give many tasks for the theory-vs-behavior comparison, with difficulty graded by how related the two categories are. The course brief explicitly asks for harder tasks built from less clearly related examples.
- **Exp 2, crossed domains:** items vary on two binary attributes in a 2×2 design, so the same items can be grouped two ways. Incongruent queries (where the two rules disagree) are the diagnostic condition.

## Tasks
- [ ] **Exp 1 inventory:** choose a category source (candidate: THINGS high-level categories, plus finer subcategories for hard pairs). Target ≥ 30 usable categories (hundreds of pairs), with ≥ 20 (ideally 40+) items per category after filtering. Define a relatedness measure for pairs (e.g. a shared superordinate, or embedding distance) to describe difficulty.
- [ ] **Exp 2 domains:** choose 3–5 crossed domains. Starting candidates (refine or replace):

  | Domain | Attribute A | Attribute B |
  |---|---|---|
  | Animals | mammal / bird | domestic / wild |
  | Foods | fruit / vegetable | red / green |
  | Countries | Europe / Asia | coastal / landlocked |
  | Numbers (synthetic) | even / odd | < 50 / > 50 |
  | Words | semantic category | first letter |

  Each cell needs ≥ 10 (ideally 20+) items, balanced across cells.
- [ ] **Attribute words** for the J-lens rule readout (e.g. "mammal", "bird", "wild", "pet", "fruit", "red"), plus 1–3 single-token synonyms each. Single-token status is checked in T-012.
- [ ] **Label sets:** ≥ 2 pairs of arbitrary single-token labels (e.g. foo/bar; others chosen after the tokenization check); a meaningful-label control; flipped labels (optional, stretch).
- [ ] **Schema:** `configs/domains.yaml` (domain → attributes, values, attribute words, notes) and `data/items/<domain>.csv` with columns `item, <attrA>, <attrB>, source, confidence, notes`. Exp 1 uses `data/items/categories.csv` with `item, category, subcategory, source, confidence`.
- [ ] Log the domain choices and rationale as **D-007** in DECISIONS.

## Acceptance criteria
- [ ] Schema files exist with ≥ 5 example rows per domain/category.
- [ ] D-007 written: domains, attributes, why each is objective and roughly independent of the other attribute.
- [ ] Status → `review`: the student signs off on the domains (a scientific choice).

## Outputs
- `configs/domains.yaml`, `data/items/*.csv` (seeded), DECISIONS D-007.

## Notes & pitfalls
- **Ground truth must be objective.** Avoid debatable items (tomato as fruit; bat "size"). Avoid attributes that change over time (living/dead).
- **Avoid confounds:** attribute B shouldn't be nearly predictable from attribute A, word frequency or token length.
- The **numbers domain** is a knowledge-free control: every item's attributes are certain, so failures are pure ICL failures.
- Surface vs semantic (first letter vs category) tests a different kind of regrouping. It's useful but may behave very differently; keep it as one domain, not the core.

## References
- PLAN §4.2–4.3, §4.7; Hebart et al. 2019 (THINGS); McRae et al. 2005; Devereux et al. 2014; Binder et al. 2016

## Work log
- 2026-10-26 (claude): Drafted the design: `configs/domains.yaml` (Exp 1 settings; 4 Exp 2 domains with attributes, values and single-token attribute-word candidates; 3 arbitrary label sets plus meaningful labels) and seeded `data/items/*.csv` via `scripts/build_item_banks.py` (see T-011). D-007 written. Main changes from the plan: **foods (fruit/vegetable × red/green) dropped** because colour isn't objective; Exp 1 categories curated rather than taken from THINGS. Relatedness measure = shared superordinate (an embedding-distance measure can be added once neutral reps exist, T-021).
  - **Needs the student's sign-off** (status `review`): the domain list, especially animals × domestication (only 6 unambiguous domestic birds), and whether curated Exp 1 categories are acceptable vs THINGS.
