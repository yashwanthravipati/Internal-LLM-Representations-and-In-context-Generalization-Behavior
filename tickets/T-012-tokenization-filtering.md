---
id: T-012
title: Tokenization filtering (items, labels, attribute words)
status: review
phase: 1
priority: P0
depends_on: [T-011, T-003]
owner: claude
estimate: S
tags: [tasks, data, tokenization]
updated: 2026-10-26
---

# T-012 — Tokenization filtering (items, labels, attribute words)

## Context
Three tokenization constraints:
1. **Labels** must be single tokens, so scoring at the decision position is clean.
2. **Attribute words** for the J-lens rule readout must be single tokens, because lens concepts are per-token (`docs/background/jacobian_lens.md`).
3. **Items** are ideally single tokens. Multi-token items can be read at their last subtoken, but token count becomes a covariate.

## Tasks
- [ ] For each model's tokenizer, count the tokens in `" " + item` (with the leading space, as items appear after a newline in prompts; confirm the exact context in which items appear in the template).
- [ ] Decide the policy, **D-008**. Default proposal: prefer single-token items; allow multi-token items represented by their last subtoken; store `n_tokens`; run a sensitivity analysis on the single-token subset.
- [ ] Check that label tokens and attribute words are single tokens; pick replacements where they aren't.
- [ ] Check that the separator (`->` vs `:` vs `→`) tokenizes the same way after different items (no merges with item tokens).
- [ ] Write `data/items/token_info_<model>.parquet` (item, n_tokens, token ids).
- [ ] Report counts per category/cell after filtering under each policy.

## Acceptance criteria
- [ ] Per-model token tables exist for all candidate models.
- [ ] D-008 logged; the label sets and attribute words are confirmed single-token for every model used.
- [ ] Post-filter counts reported; cells below the minimum are flagged back to T-011.

## Outputs
- `src/icl_geometry/tasks/tokenization.py`, `data/items/token_info_<model>.parquet`, DECISIONS D-008.

## Notes & pitfalls
- Qwen tokenizers differ in leading-space handling. Test the exact string the model sees in context, not the bare word.
- If one tokenizer is shared across the Qwen sizes, one table may serve them all, but verify.

## References
- PLAN §4.7; `docs/background/jacobian_lens.md` (single-token limitation)

## Work log
- 2026-10-26 (claude): `src/icl_geometry/tasks/templates.py` (shared `Template.render` returning the query-item char span; templates `arrow` and `io`), `src/icl_geometry/tasks/tokenization.py` (in-context item tokenization via offsets; boundary and suffix checks; `single_token_id`), `scripts/check_tokenization.py` → `data/items/token_info_qwen3.5-0.8b.parquet`, `token_info_qwen3.5-2b-base.parquet`, `results/tokenization/REPORT_<model>.md`. Tests in `tests/test_tokenization.py`.
  - Results (identical for both models; shared vocabulary): no unclean boundaries or separator changes. Single-token items: `arrow` 22% (categories 220/970), `io` 53% (515/970; countries 71/84). Numbers are 2 tokens under both templates.
  - Fixed label/attribute words: `dax/wug` → `dax/mip`; " mammal" and " landlocked" aren't single tokens (replaced, see D-008). All label sets, meaningful labels and attribute words are now confirmed single-token.
  - Post-filter counts: under the D-008 policy (multi-token allowed), no items are removed, so counts equal T-011's. Under a strict single-token policy many cells fall below minimum (e.g. `arrow` animals 22/96), which is why D-008 allows multi-token items.
- 2026-10-26 (claude): Work complete and verified; status held at `review` only because a dependency (T-002 git init, or T-011 item spot-check) is still open. Flip to `done` once that closes.
