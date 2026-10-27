# report/

The final report (T-061) lives here. **The student owns the text.** Agents may draft sections, but every claim and number gets the student's review before submission.

## Files

| File | Status |
|---|---|
| `report.md` | Draft skeleton (2026-10-27). Introduction and Methods are first drafts. Results contain **only placeholders**. Discussion is a scaffold of prompts. |

## What is real and what is a placeholder

- **Introduction, Methods:** written from `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/background/` and the implemented code in `src/icl_geometry/` and `scripts/`. If the code or a decision changes, update Methods to match.
- **`[TBD: …]`:** an open decision or missing fact. The main ones:
  - T-001: course format, length and deadline;
  - D-005: lock the primary metrics;
  - D-007: domains and items;
  - D-009: the split;
  - D-010: template, m grid and label sets;
  - D-011 / D-012: neutral-representation and outlier handling;
  - T-013: the knowledge filter.
- **`> RESULT PENDING — source: results/<run_id>/…`:** a results slot. Fill it **only** from a test-split run directory of the main model, and cite that directory. Pilot and smoke-test outputs (0.8B dev model, `results/*smoke*`, `results/*pilot*`) are plumbing checks, not results, and must not appear in the report.
- **Figure slots** use the names the analysis scripts write:
  - `exp1_rho_by_layer`, `exp1_scatter`, `exp1_learning_curves` (`scripts/run_exp1_analysis.py`);
  - `h2_profile` (`run_exp2_h2.py`);
  - H3 AUROC and `h3_layer_profile` (`run_exp2_h3.py`);
  - `h4_readout` (`run_exp3_readout.py`);
  - H5 AUROC (`run_exp3_jspace.py`, table `h5_auroc.csv`).

  Final figures come from the figure pipeline (`scripts/make_figures.py`, `figures/manifest.yaml`; T-060), where the H3 and H5 figures are named `h3_auroc` and `h5_auroc`. Never make them by hand.

## Building

The report is Markdown for now. The final format (LaTeX, Word or PDF, and the page limit) depends on T-001. To make a PDF from Markdown, for example: `pandoc report/report.md -o report/report.pdf`. Pandoc is not a project dependency.

## Before submitting

- Every number cites its run directory, in the text or an appendix table.
- No placeholder or `[TBD]` remains.
- Claims don't go beyond the evidence: decodable ≠ used, and correlational ≠ causal unless T-044 was done. Nulls are stated plainly.
