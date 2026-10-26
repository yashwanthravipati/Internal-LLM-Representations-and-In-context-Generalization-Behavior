---
id: T-060
title: Figure pipeline & shared style
status: todo
phase: 6
priority: P1
depends_on: [T-005]
owner: unassigned
estimate: S
tags: [writing, figures, reproducibility]
updated: 2026-10-25
---

# T-060 — Figure pipeline & shared style

## Context
Every report figure must be regenerated from `results/` by a script: no hand-made or notebook-only figures. Set this up early so analysis tickets plug into it.

## Tasks
- [ ] `src/icl_geometry/plotting.py`: a shared matplotlib style (font sizes, colorblind-safe palette, consistent colors for cued/uncued, congruent/incongruent, and model sizes), with layers on **normalized depth (0–100)**.
- [ ] `scripts/make_figures.py`: one function per report figure, reading from the run directories named in a small manifest (`figures/manifest.yaml`), writing PNG + PDF to `figures/`.
- [ ] `figures/README.md`: figure index (figure → script function → source run → ticket).

## Acceptance criteria
- [ ] `uv run python scripts/make_figures.py` regenerates every figure in the manifest from scratch.
- [ ] The style is used by the analysis tickets (T-024, T-032, T-033, T-041, T-043).

## Outputs
- `src/icl_geometry/plotting.py`, `scripts/make_figures.py`, `figures/manifest.yaml`, `figures/README.md`.

## Notes & pitfalls
- Label figures produced from placeholder or synthetic data unmistakably, so they can never slip into the report.

## References
- PLAN §7.1; CLAUDE.md (guardrails)

## Work log
