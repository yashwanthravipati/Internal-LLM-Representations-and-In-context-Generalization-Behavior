---
id: T-020
title: Activation extraction pipeline
status: in-progress
phase: 2
priority: P0
depends_on: [T-003, T-005]
owner: claude
estimate: M
tags: [models, activations, infra]
updated: 2026-10-26
---

# T-020 — Activation extraction pipeline

## Context
Every geometric analysis needs residual-stream activations at specific token positions and layers (PLAN §5, D-006). One robust extractor serves neutral representations (T-021), in-context representations (T-030) and the J-lens analyses (T-040+).

## Tasks
- [ ] `extract(model, prompts, positions, layers) → cached arrays` (T-005 layout: fp16 memmap `[n_prompts, n_layers+1, d_model]` plus `index.parquet`).
- [ ] Positions:
  - `query_item`: the last subtoken of the query item;
  - `decision`: the final prompt token;
  - optional `demo_items`, stored separately with a ragged index.
- [ ] Layers: embeddings plus the output of every block. **Document precisely** which tensor is used: `output_hidden_states` vs forward hooks, and whether the last entry has the final norm applied. **Match the convention `jlens` uses** (from T-004), so lens application on cached activations is valid.
- [ ] Position bookkeeping that is robust to padding and multi-token items (use the offset mapping, or token spans tracked during prompt construction).
- [ ] Batched, streaming to disk, resumable (skip rows already written).
- [ ] CLI: `uv run python scripts/run_extract.py --config configs/runs/<name>.yaml`.
- [ ] Tests:
  - activations for a prompt are identical (within tolerance) alone vs in a batch;
  - decoding the token at each recorded position gives the expected item or separator token;
  - layer count and shape are correct.

## Acceptance criteria
- [ ] Extraction runs on the dev model for 1k prompts; the tests pass.
- [ ] Disk usage and time per 1k prompts are logged in `docs/COMPUTE.md`.
- [ ] The hook-point convention is documented in the module docstring and in `docs/background/jacobian_lens.md`.

## Outputs
- `src/icl_geometry/models/activations.py`, `scripts/run_extract.py`, `tests/test_activations.py`.

## Notes & pitfalls
- Store fp16 but compute geometry in fp32/fp64. Covariance eigenvalues are sensitive to precision.
- If storage is tight, store every other layer for the big sweeps and all layers for a subset. Make the layer list part of the cache key.

## References
- PLAN §5, §7.3–7.4; D-006

## Work log
- 2026-10-26 (claude): Extraction is shared with scoring: `behavior/scoring.run_forward` (right-padded batches sorted by length; hooks on every block; states at `query_item` = last subtoken of the query span located via offsets, and `decision` = last real token), `models/activations.extract_states` for arbitrary texts, and `scripts/run_behavior.py` with `cache_activations: true` writing the T-005 cache (`data/cache/<model>/<prompt_set_hash>/<position>.npy` + `index.parquet`, written in `.partial/` and renamed). Convention documented in module docstrings and `docs/background/jacobian_lens.md`: states[0] = embeddings, states[l+1] = block l output (pre-final-norm for the last block, matching jlens).
  - Tests: `tests/test_activations.py` (recorded position decodes to the item's last subtoken; shapes [n_layers+1, d]; alone vs batch) and `tests/test_scoring.py` (padding invariance of states and margins). All pass.
  - Not done: demo-item positions (ragged), resumability, and the 1k-prompt timing (to be logged from the T-030 run). A separate `scripts/run_extract.py` wasn't needed (run_behavior covers it).
