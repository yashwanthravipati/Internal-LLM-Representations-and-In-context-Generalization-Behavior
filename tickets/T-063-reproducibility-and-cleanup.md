---
id: T-063
title: Reproducibility pass & cleanup
status: todo
phase: 6
priority: P1
depends_on: [T-061]
owner: unassigned
estimate: S
tags: [reproducibility, infra]
updated: 2026-10-25
---

# T-063 — Reproducibility pass & cleanup

## Context
The final code submission (if the course requires one), and the project's credibility, depend on someone being able to reproduce the headline results from a fresh clone.

## Tasks
- [ ] On a fresh clone, follow the README to reproduce: environment → prompts → behavior → activations → Exp 1 analysis → figures (on the dev model if compute is limited). Fix anything that breaks.
- [ ] README: a full "Reproduce the results" section with commands and approximate runtimes.
- [ ] Pin everything (uv.lock, the jlens commit, model revisions/hashes, lens file revisions).
- [ ] Remove dead code and stale notebooks (or move them to `notebooks/archive/`); ensure `ruff` and `pytest` pass.
- [ ] Archive the `results/*/REPORT.md` summaries; close or `wontfix` the remaining tickets with notes; regenerate the board.

## Acceptance criteria
- [ ] A fresh-clone reproduction of the Exp 1 headline figure succeeds, following only the README.
- [ ] `python3 scripts/tickets.py check` passes; no ticket is left `in-progress`.

## Outputs
- Updated README, pinned environment, a clean repo.

## References
- PLAN §7; CLAUDE.md (traceability)

## Work log
- 2026-10-27 (claude): Early pieces, ahead of the dependency: `.github/workflows/tests.yml` (uv sync, ruff, fast tests, ticket check on push/PR; not yet run on GitHub) and a `Makefile` wrapping the RUNBOOK stages (`make help`). Fixed a `RunConfig.from_yaml` bug: reloading a saved `config.yaml` nested `extra` under `extra["extra"]`, so options such as `eval_split` or `limit` were invisible to scripts that re-read a run's config (`check_integrity.py` silently ignored `limit`). Regression test in `tests/test_io.py`.
