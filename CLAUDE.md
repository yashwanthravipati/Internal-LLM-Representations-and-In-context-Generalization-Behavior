# CLAUDE.md — agent guide for this repo

Read this first. It gives any agent (or human) working here the big picture, the class context, and the rules of the road. Detailed design lives in `docs/PLAN.md`; work items live in `tickets/`.

## What we're building, and why

This is the **final project for Neuro 240 / Phys 240: Biological and Artificial Intelligence (Fall 2026)** by Yashwanth Ravipati. It is **Project 5** from the course's official list: *Can the geometry of internal LLM representations predict in-context generalization behavior?*

LLMs can learn a categorization from a few examples in the prompt (`dog → foo, cat → foo, oak → bar, fern → bar, bat → ?`). A neuroscience theory (Sorscher, Ganguli & Sompolinsky 2022, *PNAS*), validated on macaque IT cortex and deep vision nets, predicts the few-shot error of a simple prototype learner from the geometry of its representations. **We test whether that theory, and representational geometry more broadly, predicts LLM in-context learning.** Two extensions:

1. **Context-dependent regrouping:** items that can be grouped two ways (mammal/bird vs domestic/wild), with the demonstrations deciding which grouping applies. Do the representations shift toward the demonstrated grouping, at which layers, and does the shift predict accuracy on unseen items? This pits two neuroscience accounts against each other: selection in the readout (Mante 2013) vs compression of the irrelevant dimension (Flesch 2022).
2. **Anthropic's Jacobian lens** (Gurnee et al. 2026): read out which concepts the model is poised to verbalize (does "mammal" light up?), and test whether geometry in the lens's J-space predicts generalization better than full hidden states, against dimension-matched baselines.

The deliverable is a research report (and probably a presentation). Format and deadlines are TBD in ticket T-001. **The scientific output is the point: clean experiments, honest results, and a clear link between neuroscience and AI.**

## The class: why it shapes the work

- **Theme: AI and neuroscience.** Every analysis should connect to a neuroscience idea, and the write-up must make that link explicit. See `docs/background/neuro_connections.md`.
- **Project 5 is pre-approved** (listed difficulty 5/10; our extensions take it to roughly 7). Course rules: the project can't duplicate another class's project or a lab project. Details and the verbatim brief: `docs/course/COURSE_CONTEXT.md`; the original list is `docs/course/Projects_List_2026.pdf`.
- **Compute:** plan for about **one 16 GB GPU** (the course's guidance for similar projects) until T-001 confirms otherwise.
- **Deadlines:** TBD (T-001). Until they're known, protect the MVP: **Experiment 1 must land before the extensions get serious effort.**

## Hypotheses and experiments (short form; full version in PLAN §2 and §4)

| Exp | Question | Hypotheses | Phase / tickets |
|---|---|---|---|
| 1 (MVP) | Does Sorscher-theory error from *neutral* item representations predict LLM few-shot error across category pairs and shot counts m? | H1 | Phase 2 · T-020–T-024 |
| 2 | Same items, different demonstrated grouping: do representations shift (H2a) or does the readout select (H2b)? Does shift predict held-out accuracy (H3)? | H2a/H2b, H3 | Phase 3 · T-030–T-034 |
| 3 | Do cued-attribute words appear in the J-lens readout (H4)? Does J-space geometry beat full hidden states at matched dimension (H5)? | H4, H5 | Phase 4 · T-040–T-045 |
| 4 (stretch) | Ambiguous demos (default grouping); scaling across model sizes | — | Phase 5 · T-050–T-051 |

## Where things are

| Path | What |
|---|---|
| `docs/PLAN.md` | **Source of truth** for the research design, analyses, architecture, phases, risks |
| `docs/DECISIONS.md` | Decision log; add an entry for any design choice |
| `docs/course/` | Course context, verbatim Project 5 brief, the project-list PDF |
| `docs/background/` | Theory notes (Sorscher equations, verified), Jacobian-lens notes, neuroscience connections |
| `tickets/` | Ticket system: `README.md` (how it works), `BOARD.md` (generated), `T-NNN-*.md` |
| `scripts/tickets.py` | Ticket command-line tool (stdlib only) |
| `src/icl_geometry/` | Python package (T-002 onward); layout in PLAN §7.1. Run order: `docs/RUNBOOK.md` |
| `configs/`, `data/`, `results/`, `figures/`, `tests/`, `notebooks/` | Created by T-002/T-005 (see PLAN §7) |

## How to work in this repo

1. **Orient:** read this file, then the PLAN sections and background docs your ticket references.
2. **Pick work:** `python3 scripts/tickets.py ready` lists todo tickets whose dependencies are closed. If the user assigned a ticket, do that one. Don't start P2 (stretch) tickets before milestone M2 (T-024 done) unless the user asks.
3. **Claim it:** `python3 scripts/tickets.py set T-0NN status in-progress owner <your-name>`.
4. **Stay in scope:** do what the ticket says. If you find new work, create a ticket (`python3 scripts/tickets.py new "Title" --phase N --depends T-0NN`) instead of silently expanding scope. If the ticket is wrong or outdated, say so and fix the ticket.
5. **Log as you go:** append dated entries to the ticket's **Work log** (what you did, where the outputs are, any surprises). Record design choices in `docs/DECISIONS.md`. If a choice changes the design, update `docs/PLAN.md` too.
6. **Close it:**
   - Set `done` only when every acceptance criterion is verified (code run, tests passing, outputs exist).
   - Set `review` when a criterion needs the user's judgment: scientific choices, domain/item curation, interpreting results.
   - Set `blocked` (and say why in the Work log) when stuck on something external.
   - The board regenerates automatically on `set`/`new`; run `python3 scripts/tickets.py check` before finishing.
7. **Report honestly to the user:** what was done, what was verified, what wasn't, and any result that contradicts the plan.

## Scientific guardrails (non-negotiable)

- **Never fabricate, simulate or "fill in" results.** If a run fails or a number is unavailable, say so. Placeholder figures must be clearly labelled as such and never reach the report.
- **Null and negative results are valid findings.** Report them plainly; don't tune the analysis until something "works".
- **Held-out discipline:** splits are fixed before analysis (D-009). Choose layers and hyperparameters on validation data only; headline numbers come from the test split.
- **Primary metrics are pre-specified** (PLAN §6.9). A change after seeing data needs a DECISIONS entry and gets labelled exploratory.
- **Watch for circularity:** late-layer decision-position states basically *are* the output. Primary predictive claims rest on query-item representations and intermediate layers (PLAN §5).
- **Controls aren't optional:** random-label demonstrations, counterbalanced label mappings and orders, the zero-shot knowledge filter, dimension-matched baselines for J-space comparisons.
- **Traceability:** every number in a report traces back to `results/<run_id>/` with its config, seed, git commit and metadata.

## Engineering conventions (being set up in T-002 / T-005)

- Python ≥ 3.11, `uv`, src layout (`src/icl_geometry`), `pytest`, `ruff`. `scripts/tickets.py` stays stdlib-only.
- Configs in `configs/` (YAML). Run outputs go to `results/<YYYYMMDD-HHMM>_<exp>_<model>_<desc>/` with `config.yaml` + `metadata.json`.
- Activations are fp16 memmaps in `data/cache/` (gitignored). Never commit large binaries, model weights or lens files.
- Notebooks are for exploration. Anything that feeds the report lives in `src/` + `scripts/` and is reproducible from the command line.
- Geometry math gets unit tests on synthetic data with known answers (e.g. Monte Carlo vs theory).
- Match the style, naming and comment density of the surrounding code.
- Git: commit only when the user asks; never push or create remotes without explicit approval.

## Key technical facts (verified 2026-10-25; details in `docs/background/`)

- **Sorscher Eq. 1:** `SNR_a = ½[‖Δx₀‖² + (R_b²/R_a² − 1)/m] / √(D_a⁻¹/m + ‖Δx₀·U_b‖²/m + ‖Δx₀·U_a‖²)`, with error `ε_a = H(SNR_a)` (Gaussian tail). Official code: github.com/bsorsch/geometry-fewshot-learning. Normalization conventions are subtle: port them from the official code and validate with Monte Carlo.
- **Jacobian lens:** `J_ℓ = E[∂h_final/∂h_ℓ]`; readout `softmax(W_U · norm(J_ℓ h))`; concept vector `v_t = J_ℓᵀ W_U[t]`. The J-space is a sparse non-negative combination of ≤ 25 lens vectors, coherent only in middle layers, and explains ≤ ~10% of variance. Package: `jlens` from github.com/anthropics/jacobian-lens (unmaintained, so pin the commit). Concepts must be single tokens.
- **Pre-fitted lenses:** HF `neuronpedia/jacobian-lens` has plain J-lens fits (`*/jlens/`) as well as **J++** variants. We use the plain fits: `qwen3.5-2b-pt/jlens` (= Qwen/Qwen3.5-2B-Base, main) and `qwen3.5-0.8b/jlens` (dev); see D-003 and T-004. A lens is valid only for the exact checkpoint it was fitted on.

## Glossary

- **ICL:** in-context learning (learning from demonstrations in the prompt, with no weight updates).
- **m / shots:** demonstrations per category.
- **Cued / uncued attribute:** the attribute the demos define vs the other one (Exp 2).
- **Congruent / incongruent query:** both rules give the same label / the rules disagree. Incongruent queries are diagnostic.
- **Neutral representation:** an item's hidden state with no demonstrations (carrier templates).
- **`query_item` / `decision` positions:** the last subtoken of the query item / the final prompt token where the label is predicted.
- **SNR, D, R, signal–noise overlap:** Sorscher geometry terms (see `docs/background/sorscher_theory.md`).
- **ΔSNR / shift score:** change in the cued-attribute geometry in context vs control (PLAN §6.4).
- **CCGP:** cross-condition generalization performance (Bernardi 2020). **RSA:** representational similarity analysis.
- **J-lens / J-space / concept vector:** see `docs/background/jacobian_lens.md`.
- **MVP / M2:** Experiment 1 complete (T-024).
