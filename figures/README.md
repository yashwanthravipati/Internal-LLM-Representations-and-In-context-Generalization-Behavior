# Figures

Every report figure is regenerated from saved run artifacts by `scripts/make_figures.py`
(no model needed). Nothing here is hand-made or notebook-only.

```bash
uv run python scripts/make_figures.py                                   # report figures
uv run python scripts/make_figures.py --only exp1_scatter               # one figure
uv run python scripts/make_figures.py --manifest figures/manifest_smoke.yaml   # plumbing check
```

- `manifest.yaml` lists the report figures and the `results/<run_id>` directories they read.
  Entries whose runs are still placeholders print `PENDING` and are skipped. Fill in the run
  paths as the real runs finish.
- Each figure is written as `<name>.png` + `<name>.pdf`, with a `<name>.source.json` sidecar
  recording the source run IDs, their git commits and the manifest entry.
- `status: smoke` figures go to `figures/smoke/` (gitignored) with a **SMOKE TEST: NOT A
  RESULT** watermark. `status: exploratory` figures get an EXPLORATORY corner label.
- Shared style: `src/icl_geometry/plotting.py` (`apply_style`, `ROLE_COLORS`, `depth`).
  Layers are shown as normalized depth: 0 = embeddings, 100 = output of the last block.

## Index

| Figure | Function | Source run (script) | Ticket | Status |
|---|---|---|---|---|
| `theory_vs_prototype_synthetic` | `theory_vs_prototype_synthetic` | `scripts/validate_theory_synthetic.py` | T-022 | made (synthetic data by design) |
| `theory_vs_prototype_neutral` | `theory_vs_prototype_neutral` | `scripts/validate_theory_neutral.py` (main model) | T-023 | pending |
| `exp1_rho_by_layer` | `exp1_rho_by_layer` | `scripts/run_exp1_analysis.py` | T-024 | pending |
| `exp1_scatter` | `exp1_scatter` | `scripts/run_exp1_analysis.py` | T-024 | pending |
| `exp1_learning_curves` | `exp1_learning_curves` | `scripts/run_exp1_analysis.py` | T-024 | pending |
| `h2_profile` | `h2_profile` | `scripts/run_exp2_h2.py` | T-032 | pending |
| `h3_auroc` | `h3_auroc` | `scripts/run_exp2_h3.py` | T-033 | pending |
| `h4_readout` | `h4_readout` | `scripts/run_exp3_readout.py` (J-lens + `--logit-lens`) | T-041 | pending |
| `h5_auroc` | `h5_auroc` | `scripts/run_exp3_jspace.py` | T-043 | pending |

The analysis scripts also save quick-look figures inside their own run directories; those are
for checking a run, not for the report.
