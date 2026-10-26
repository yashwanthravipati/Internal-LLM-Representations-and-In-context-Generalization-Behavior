---
id: T-002
title: Repo scaffolding & Python environment
status: review
phase: 0
priority: P0
depends_on: []
owner: claude
estimate: S
tags: [infra]
updated: 2026-10-26
---

# T-002 — Repo scaffolding & Python environment

## Context
So far the repo holds only docs, tickets and `scripts/tickets.py`. Every other ticket needs a package skeleton, a reproducible environment, and test and lint tooling. Target layout: PLAN §7.1.

## Tasks
- [ ] `git init` if the repo isn't a git repo yet. **Ask the user before the first commit**, and never add a remote or push without approval.
- [ ] `pyproject.toml` managed with `uv` (Python ≥ 3.11); package `icl_geometry` under `src/`.
- [ ] Runtime dependencies: torch, transformers, accelerate, safetensors, numpy, scipy, pandas, pyarrow, scikit-learn, statsmodels, matplotlib, tqdm, pyyaml. Dev dependencies: pytest, ruff, ipykernel. (`jlens` is added in T-004 from GitHub at a pinned commit.)
- [ ] Skeleton: `src/icl_geometry/{tasks,models,behavior,geometry,lens,analysis}/__init__.py`, `src/icl_geometry/io.py` (stub), `configs/`, `data/items/`, `data/cache/`, `results/`, `figures/`, `notebooks/`, `tests/`.
- [ ] `.gitignore`: `data/cache/`, `data/lenses/`, `results/*` except `results/**/REPORT.md`, `*.pt`, `*.npy`, `*.safetensors`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, local HF caches.
- [ ] ruff config (line length 100) and pytest config in `pyproject.toml`; one trivial test.
- [ ] README: add an "Environment setup" section (`uv sync`, `uv run pytest`). If the course cluster needs conda/modules, document that path too.

## Acceptance criteria
- [ ] On a fresh clone: `uv sync` succeeds; `uv run pytest` passes; `uv run python -c "import icl_geometry"` works.
- [ ] `uv run ruff check .` passes.
- [ ] `python3 scripts/tickets.py check` still passes (the script stays stdlib-only).
- [ ] `uv.lock` committed (once the user approves commits).

## Outputs
- `pyproject.toml`, `uv.lock`, `.gitignore`, `src/icl_geometry/…`, `tests/test_smoke.py`, updated README.

## Notes & pitfalls
- Pin a torch build that matches the CUDA version found in T-003. If T-003 hasn't run yet, use a CPU-compatible default and revisit.
- Don't add heavy optional dependencies (TransformerLens, nnsight) unless a ticket needs them. Plain HF hooks are enough for this project.

## References
- PLAN §7.1, §7.5

## Work log
- 2026-10-26 (claude): Created `pyproject.toml` (uv, hatchling, Python ≥ 3.11; ruff line length 100; pytest with a `slow` marker), `.gitignore`, `src/icl_geometry/{tasks,models,behavior,geometry,lens,analysis,utils}/`, `configs/{runs,prompts}/`, `data/items/`, `results/`, `figures/`, `notebooks/`, `tests/test_smoke.py`, README "Environment setup" section. Verified: `uv sync` OK (torch 2.14.1, transformers 5.20.2, Python 3.12.15), `uv run pytest` passes, `import icl_geometry` works, `uv run ruff check .` passes (fixed two lint issues in `scripts/tickets.py`, behaviour unchanged), `tickets.py check` passes.
  - **Not done:** `git init`. The agent sandbox forbids creating `.git/` in this folder, so the user must run it (see the session summary). `uv.lock` exists but is uncommitted. Status `review` until the repo is initialised and the first commit is approved.
  - Note: inside the agent sandbox, MPS (Apple GPU) is unavailable ("MPS backend is supported on macOS 14.0+" — a sandbox artefact; the host is macOS 26 on an M5 Pro). Agent runs use CPU; see `docs/COMPUTE.md` (T-003).
