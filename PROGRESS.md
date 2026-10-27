# Progress report

_Last updated 2026-10-27. The tickets (`tickets/`, `python3 scripts/tickets.py list`) and
`docs/DECISIONS.md` hold the full detail; this page is the overview._

## Where things stand

The experimental infrastructure for all three experiments is built, unit-tested and
smoke-tested end to end on the 0.8B development model. **No real experiment has been run on the
main model (Qwen3.5-2B-Base), and there are no results yet.** Every number below comes from a
smoke test or a math check on the dev model, never from a hypothesis test.

What blocks real results, in order:

1. **Your decisions** (no compute needed):
   - D-007 (domains) and the T-011 item spot-check;
   - D-009 (40/60 val/test split);
   - T-001 (course deadlines and format).
2. **The 2B-Base pilot** (T-016), then **D-010** (template, m grid, label sets).
3. **The main runs** in `docs/RUNBOOK.md` order. They need a GPU or Apple MPS outside the sandbox.

## Done in this session (2026-10-27, no model runs)

### Figure pipeline (T-060)

Every report figure now comes from a script that reads saved run outputs. No figure is
hand-made or built in a notebook.

- **What was built:**
  - `scripts/make_figures.py`: 9 figure functions, covering the theory validations, Exp 1 (ρ by layer, scatter, learning curves), H2, H3, H4 and H5.
  - `figures/manifest.yaml`: maps each figure to the `results/<run_id>` directories it reads.
  - Each figure is written as PNG + PDF, plus a `.source.json` file recording the source run IDs and git commits.
- **Guardrail built in:** entries marked `status: smoke` go to the gitignored `figures/smoke/` with a diagonal "SMOKE TEST: NOT A RESULT" watermark, so they can't drift into the report (D-014). Entries whose runs don't exist yet print `PENDING`.
- **Shared style** (`src/icl_geometry/plotting.py`): fixed colors per role (cued/uncued, conditions, representations, models), and layers on a normalized depth axis (0–100).
- **Verified:**
  - All 9 figures render from the 0.8B smoke runs, and I checked them for layout problems.
  - `tests/test_figures.py` renders every figure from synthetic run directories, so CI covers it without real data.
- **Your step:** after the real runs, paste their run paths into `figures/manifest.yaml` and run `make figures`.

### Theory vs simulation on real LLM states (T-023)

- **What it does:** `scripts/validate_theory_neutral.py` compares the Sorscher Eq. 1 prediction with Monte Carlo prototype and exemplar learners. It uses cached neutral item representations and runs no model.
- **Dev-model math check** (0.8B; `results/20261027-1509_t023_qwen3.5-0.8b_dev/`; not a result). Setup: 300 category pairs × 5 depths × m ∈ {1, 2, 4, 8}.
  - Theory and simulation agree well: Spearman ρ 0.85–0.99 and mean absolute gap 0.028.
  - At the embeddings, the theory underestimates error by 0.01–0.04.
  - At later layers with m = 1–2, it overestimates by 0.01–0.04.
  - For m ≥ 4 the gap is at most 0.015.
  - So the port of the theory behaves sensibly on real LLM geometry, not only on synthetic Gaussians.
- **Next:** rerun on 2B-Base once its neutral cache exists.

### Exp 2 controls and robustness (T-034, code only)

- **What was built:**
  - `src/icl_geometry/analysis/robustness.py`, `scripts/run_exp2_robustness.py` and `tests/test_robustness.py`.
  - The script produces `ROBUSTNESS.md`: a table of claims (H2 ΔSNR_cued, H2 selectivity, H3 shift AUROC) × variations, with each cell marked holds / weakens / fails / n/a.
  - Variations covered: label set, label mapping (counterbalancing), demo order, template, m, single carriers instead of the carrier mean (D-011), and outlier handling (D-012).
- **New permutation null** for ΔSNR: main/random-label assignments are shuffled within each group.
- **Classification rules** are proposed in **D-015** and need your sign-off.
- **Tests:** a planted effect present only for label_map 0 is classified correctly.
- **Smoke run:** `results/20261027-1510_exp2robust_qwen3.5-0.8b_smoke/` (plumbing only).
- **Still needs model runs:** the second-model-size check.

### Report skeleton (T-061)

- **`report/report.md`:**
  - Introduction and Methods are first drafts, checked against the code as implemented.
  - Results are `RESULT PENDING` slots naming the run directory and figure each will come from. They contain no numbers.
  - Discussion is a scaffold of prompts, including how to frame nulls.
  - Open decisions are marked `[TBD: …]`.
- **`report/README.md`:** explains what is real and what is a placeholder. You own the final text.

### Reproducibility (part of T-063)

- **`.github/workflows/tests.yml`:** runs lint, fast tests and the ticket check on every push or PR. It hasn't run on GitHub yet: it's uncommitted, and pushing workflow files may need a token with the `workflow` scope.
- **`Makefile`:** shortcuts for every RUNBOOK stage (`make help`). Targets that need a model are marked.

### Fixes and doc cleanup

- **Bug fixed:** `RunConfig.from_yaml` nested `extra` inside `extra` when it reloaded a saved run config. As a result, scripts that re-read a run's config couldn't see options such as `eval_split`, and `check_integrity.py` silently ignored `limit`. A regression test is in `tests/test_io.py`.
- **New option:** `scripts/run_exp1_analysis.py` takes `--desc` (e.g. `smoke`). I ran it on the cached 0.8B pilot data as a plumbing check: `results/20261027-1507_exp1_qwen3.5-0.8b_smoke/`.
- **Doc updates:**
  - `docs/RUNBOOK.md` covers the new scripts.
  - `CLAUDE.md` no longer says the package is "not created yet", and describes the plain J-lens fits now used (D-003).
  - D-007 now lists dax/mip instead of dax/wug.

### Verification

- `ruff check src scripts tests`: clean.
- `pytest -m "not slow"`: 74 passed.
- `python3 scripts/tickets.py check`: ok (36 tickets).
- **Nothing has been committed or pushed.** Say the word and I'll commit in logical chunks under your account, without the Claude trailer.

## Things that need your judgment

**Decisions waiting on you:**
- D-007 domains;
- D-009 split;
- D-011 neutral carriers;
- D-012 outlier handling;
- D-015 robustness rules;
- the T-011 item spot-check;
- T-001 course logistics;
- D-005: lock the primary metrics before Exp 1 runs on test.

**Places where PLAN and the code disagree.** The report helper found these; none were changed, because each is a design call.

1. **Splits:** PLAN §6.8 mentions a train/val/test split. D-009 and the code use val/test only (40/60).
2. **Decoder:** PLAN §6.4 says "cross-validated linear probe". The code defaults to a nearest-class-mean decoder, because logistic probes overfit item identity; logistic is still available.
3. **Concept vector formula:** PLAN, CLAUDE.md and `jacobian_lens.md` give the J-lens concept vector as J_lᵀ W_U[t]. The code uses J_lᵀ(g ⊙ W_U[t]), with the RMSNorm gain g. The code version is correct for ranking tokens under the lens.
4. **Signal normalization:** PLAN §6.2 normalizes the signal by R_a². The code uses tr(C_a), following the official code and documented in `sorscher_theory.md`.
5. **H3 baseline (the most substantive one):** H3 is phrased as "beyond neutral geometry", but the H3 baseline is only label bias + m. Neutral SNR, word frequency and layer-0 geometry are not in it. Adding them before the real runs would match PLAN §6.6.
6. **H5 representations:** PLAN lists sparse codes, supervised probe directions and the tuned lens. The H5 script covers full, J-space, the attribute span, the logit-lens span, PCA and random subspaces.
7. **Mixed effects:** `prediction.mixed_effects` exists, but no script calls it.
8. **Exp 2 size:** PLAN §4.6 assumes 4 values of m for Exp 2, but the m grid is {2, 4, 8}.
9. **Git commit not recorded:** inside this sandbox there's no `.git`, so run metadata records `git.commit: null`. Real runs from your own clone record it properly.

**Other notes:**
- The robustness script's H2 layer is selected on val as the maximum ΔSNR_cued at `query_item`, and on smoke data that was the last state. It isn't circular (it's the query position, not the decision readout), but you may prefer to pass `--h2-layer` or restrict selection to middle layers. If so, log it in D-015.
- The linear-SVM learning rule (an optional part of T-023) is not implemented.

## What runs where

| Needs | Stages |
|---|---|
| Nothing (CPU, seconds to minutes) | `make test`, `make figures`, `make theory-synthetic`, `make theory-neutral` (after neutral cache), all `exp2-*` analysis on cached activations |
| Model forward passes (GPU/MPS recommended) | knowledge filter (T-013), pilot (T-016), neutral reps (T-021), Exp 1/2 behavior + activation runs, H4 readout, H5 J-space (gradient pursuit) |
| Infeasible on CPU | fitting our own lens (T-045, contingency only) |
