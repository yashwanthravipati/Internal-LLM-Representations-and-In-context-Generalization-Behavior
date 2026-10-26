---
id: T-005
title: Experiment config, run IDs & results storage conventions
status: done
phase: 0
priority: P0
depends_on: [T-002]
owner: claude
estimate: S
tags: [infra, reproducibility]
updated: 2026-10-26
---

# T-005 — Experiment config, run IDs & results storage conventions

## Context
Every number in the report must trace back to a run with its config, seed and code version (CLAUDE.md guardrails). Set the conventions once, before the pipelines exist. Formats: PLAN §7.3.

## Tasks
- [ ] Config schema (YAML) for runs: model, prompt_set, positions, layers, batch_size, dtype, seed, output dir. Configs live under `configs/runs/` and `configs/prompts/`.
- [ ] Run ID convention `<YYYYMMDD-HHMM>_<exp>_<model>_<desc>`. Each run writes `results/<run_id>/config.yaml` (resolved), `metadata.json` (git commit + dirty flag, package versions, GPU, start/end time, host), and its outputs.
- [ ] Activation cache layout: `data/cache/<model>/<prompt_set_hash>/<position>.npy` (fp16 memmap `[n_prompts, n_layers+1, d_model]`) plus `index.parquet` (row → prompt_id). The prompt-set **content hash** in the path prevents stale caches.
- [ ] `src/icl_geometry/io.py`: read/write prompts (JSONL), behavior (parquet), activations (memmap with index), run metadata.
- [ ] Seeding utility (python, numpy, torch).
- [ ] Unit tests for round trips and hash stability.

## Acceptance criteria
- [ ] `io.py` with passing round-trip tests.
- [ ] One example config in `configs/runs/example.yaml`.
- [ ] PLAN §7.3 updated if the implementation deviates.

## Outputs
- `src/icl_geometry/io.py`, `src/icl_geometry/utils/seed.py` (or similar), `configs/runs/example.yaml`, `tests/test_io.py`.

## Notes & pitfalls
- Keep it simple: plain YAML plus dataclasses. Hydra is optional and probably overkill here.
- memmap writes must be flushed. Write to a temp path, then rename, so interrupted runs don't leave half-written caches that look valid.

## References
- PLAN §7.2–7.3

## Work log
- 2026-10-26 (claude): Implemented `src/icl_geometry/io.py`: `RunConfig` (YAML dataclass; unknown keys kept in `extra`), `make_run_id` (`<YYYYMMDD-HHMM>_<exp>_<model>_<desc>`), `Run` context manager (writes resolved `config.yaml` and `metadata.json` with git commit/dirty flag, package versions, device, host, start/end, status), JSONL/parquet helpers (temp-file + rename), `prompt_set_hash` (order-sensitive SHA-256 of canonical JSON, 12 hex), `ActivationWriter` (fp16 `open_memmap` per position in `<dir>.partial/`, refuses to finalize if any row is unwritten, renames on close) and `load_activations`. `src/icl_geometry/utils/seed.py` (`seed_everything`). Example config `configs/runs/example.yaml`. Tests `tests/test_io.py` (7 tests) pass; ruff clean. Implementation matches PLAN §7.3, no deviations. Note: git metadata is `null` until the repo is initialised (T-002).
- 2026-10-26 (claude): Work complete and verified; status held at `review` only because a dependency (T-002 git init, or T-011 item spot-check) is still open. Flip to `done` once that closes.
- 2026-10-26 (claude): Dependency closed; status → done.
