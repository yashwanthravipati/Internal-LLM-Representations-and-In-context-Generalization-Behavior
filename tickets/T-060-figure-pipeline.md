---
id: T-060
title: Figure pipeline & shared style
status: in-progress
phase: 6
priority: P1
depends_on: [T-005]
owner: claude
estimate: S
tags: [writing, figures, reproducibility]
updated: 2026-10-27
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
- 2026-10-27 (claude): Built the pipeline.
  - `src/icl_geometry/plotting.py`: `ROLE_COLORS` (fixed colors for cued/uncued, conditions, predictors, H5 representations, models), `depth()` (normalized depth 0–100), `watermark()` / `banner()`, and `save(..., formats=("pdf",))`.
  - `scripts/make_figures.py`: 9 figure functions (theory_vs_prototype_synthetic/neutral, exp1_rho_by_layer, exp1_scatter, exp1_learning_curves, h2_profile, h3_auroc, h4_readout, h5_auroc), each reading only saved run files. Writes PNG + PDF + a `.source.json` sidecar (source run IDs and git commits). Placeholder runs print PENDING; `status: smoke` goes to the gitignored `figures/smoke/` with a watermark (D-014).
  - `figures/manifest.yaml` (report figures; all but the synthetic validation are PENDING until real runs exist), `figures/manifest_smoke.yaml` (0.8B smoke runs), `figures/README.md` (index).
  - `tests/test_figures.py`: every figure function renders from synthetic run dirs with the analysis scripts' file/column layout; quarantine and PENDING behaviour tested.
  - Verified: `make_figures.py --manifest figures/manifest_smoke.yaml` makes all 9 figures; inspected them for layout (fixed two legend overlaps).
  - **Remaining:** fill in `figures/manifest.yaml` run paths once the real runs exist. The analysis scripts still draw their own quick-look figures with `apply_style`/`SERIES`; the report figures come from `make_figures.py`.
