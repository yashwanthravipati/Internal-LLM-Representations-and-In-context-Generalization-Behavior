# Project plan: does representational geometry predict in-context generalization?

_Last updated: 2026-10-26 · Owner: Yashwanth Ravipati · Course: Neuro 240 / Phys 240, Fall 2026 (Project 5)_

This is the source of truth for the research design and the build-out. Tickets in `tickets/` implement it. When the plan changes, edit this file **and** log the change in `docs/DECISIONS.md`.

---

## 1. Summary

LLMs can learn a categorization from a few examples in the prompt (`dog → foo, cat → foo, oak → bar, fern → bar, bat → ?`). Sorscher, Ganguli & Sompolinsky (2022, PNAS) derived a theory that predicts the few-shot error of a simple *prototype* learner from four measurable properties of a representation's geometry. They validated it on deep vision networks and on macaque IT cortex. **We ask whether that theory, and representational geometry more broadly, predicts the in-context learning (ICL) behavior of LLMs.**

We go beyond the course brief in two ways:

1. **Context-dependent regrouping.** We use items that can be grouped two ways (e.g. mammal/bird vs domestic/wild), and only the demonstrations say which grouping is correct. Layer by layer, we measure whether the representations of the *same items* reorganize toward the demonstrated grouping, and whether the strength of that reorganization predicts accuracy on unseen items.
2. **Jacobian lens.** We use Anthropic's Jacobian lens (Gurnee et al. 2026) to read out which concepts the model is "poised to verbalize" while it solves the task (does "mammal" or "wild" light up?). We then test whether geometry computed in the lens's *J-space* predicts generalization better than geometry of the full hidden state, against dimension-matched baselines.

The neuroscience framing draws on: concept manifolds in IT (Sorscher 2022), category learning in PFC (Freedman 2001), context-dependent selection in PFC (Mante 2013), compression of irrelevant dimensions (Flesch 2022), the geometry of abstraction (Bernardi 2020), and global workspace theory (the J-lens paper). See `docs/background/neuro_connections.md`.

---

## 2. Research questions & hypotheses

- **RQ1.** Does the geometry of an LLM's item representations predict how well it learns a categorization in context?
- **RQ2.** When the demonstrations regroup the same items, do the representations change, and where (which layers, which token positions)?
- **RQ3.** Does the strength of that change predict generalization to unseen items?
- **RQ4.** Do Jacobian-lens features (verbalizable concepts / J-space) carry the predictive signal better than full hidden states?

| ID | Hypothesis | Test | Falsified by |
|---|---|---|---|
| **H1** | Sorscher-theory error computed from *neutral* item representations predicts LLM ICL error across category pairs, including its dependence on the number of shots *m* | Exp 1 · T-024 | Held-out ρ(theory, LLM) not above a shuffled baseline at the layer chosen on validation data |
| **H2a** | **Representational shift:** demonstrations move query-item representations along the cued attribute and compress the uncued one (Flesch-like) | Exp 2 · T-032 | No rise in cued-attribute SNR/decodability at the query-item position vs the random-label control |
| **H2b** | **Readout selection:** both attributes stay equally decodable at the query-item position; selection happens only at the decision position (Mante-like) | Exp 2 · T-032 | Uncued attribute compressed at the query-item position |
| **H3** | Shift strength predicts held-out accuracy beyond neutral geometry | Exp 2 · T-033 | No incremental held-out predictive value |
| **H4** | The cued attribute's concept words rise in the J-lens readout in intermediate layers, more on correct trials | Exp 3 · T-041 | No difference vs uncued words or vs the random-label control |
| **H5** | J-space geometry predicts generalization at least as well as full hidden states and better than dimension-matched random/PCA subspaces | Exp 3 · T-043 | J-space ≤ matched random subspaces |

H2a and H2b compete. Mixed outcomes are expected and informative (e.g. shift in middle layers at the decision position only). **Null results are valid outcomes and will be reported as such.**

---

## 3. Background (pointers)

- `docs/background/sorscher_theory.md`: the theory, equations verified against the paper, estimation details, how it maps onto ICL, and where its assumptions may break.
- `docs/background/jacobian_lens.md`: what the J-lens is, J-space, concept vectors, available pre-fitted lenses, caveats.
- `docs/background/neuro_connections.md`: the neuroscience literature each analysis connects to.
- `docs/course/COURSE_CONTEXT.md`: the course, its rules, and the original Project 5 text.

---

## 4. Experimental design

### 4.1 Models

- **Primary family: Qwen3.5 (small, open weights).** HF `neuronpedia/jacobian-lens` hosts both J++ lenses and **original J-lens fits** (Anthropic's `jlens` code). We use the original J-lens (D-003, revised 2026-10-26 after T-004).
- **Dev model:** Qwen3.5-0.8B (post-trained; lens `qwen3.5-0.8b/jlens`). **Main:** **Qwen3.5-2B-Base** (lens `qwen3.5-2b-pt/jlens`, fitted on exactly this base checkpoint). **Optional scaling:** Qwen3.5-4B, Qwen3.5-9B-Base (lenses exist) if compute allows (T-051).
- Qwen3.5 is a hybrid architecture (3 Gated-DeltaNet linear-attention blocks per full-attention block). Residual-stream access and the lens work the same way at every block (T-003/T-004).
- Few-shot prompts use **raw text completion** (no chat template) unless T-004/T-016 show the lens-matched checkpoint needs something else.
- **Compute baseline for planning:** one ~16 GB GPU. The course list uses this for comparable projects; confirm in T-001.

### 4.2 Experiment 1: neutral geometry → few-shot ICL (core Project 5; the MVP)

- **Tasks:** binary categorization between pairs of categories (a, b), e.g. animal vs vehicle, fruit vs vegetable, bird vs insect. Many categories give hundreds of pairs and a wide spread of difficulty: semantically related categories are harder, as the brief suggests ("construct harder ones by using less clearly related examples").
- **Shots:** m ∈ {1, 2, 4, 8} examples per category. Queries are held-out items.
- **Representations:** *neutral* item representations (no demonstrations), every layer (§5).
- **Mapping to the theory:** concept manifold = category; manifold samples = item words; m = shots per category. Theory error ε_a = H(SNR_a) for each pair and direction.
- **Outcome:** LLM error on held-out queries, compared with (i) the theory's prediction and (ii) an empirical prototype classifier on the same representations.

### 4.3 Experiment 2: context-dependent regrouping

- **Crossed 2×2 domains:** items vary on two binary attributes A and B. The demonstrations alone cue A or B.
- **Example**, Animals (A = mammal/bird, B = domestic/wild):
  - Cue A: `dog → foo, wolf → foo, chicken → bar, eagle → bar`
  - Cue B: `dog → foo, chicken → foo, wolf → bar, eagle → bar`
  - **Incongruent** queries (the answer depends on the rule): *duck* (A → bar, B → foo), *tiger* (A → foo, B → bar).
  - **Congruent** queries (same answer under both rules): *cow* (foo/foo), *hawk* (bar/bar).
- Demonstrations are **balanced on the uncued attribute**, so only the cued attribute explains the labels.
- **Primary behavioral measure:** accuracy and logit margin on incongruent queries.
- **Representations:** query-item token and decision token (§5), every layer, in context. Also neutral representations of the same items, and representations under random-label demonstrations (the control).
- **Candidate domains** (finalized in T-010/T-016):

| Domain | Attribute A | Attribute B | Notes |
|---|---|---|---|
| Animals | mammal / bird | domestic / wild | Classic semantic attributes |
| ~~Foods~~ | ~~fruit / vegetable~~ | ~~red / green~~ | **Dropped (D-007):** colour isn't objective for most produce |
| Countries | Europe / Asia | coastal / landlocked | Objective ground truth |
| Numbers | even / odd | < 50 / > 50 | Synthetic control that needs no world knowledge |
| Words | animal / food | first letter b / p | Surface vs semantic grouping |

Final domain spec and item banks: `configs/domains.yaml`, `data/items/*.csv` (D-007).

### 4.4 Experiment 3: Jacobian lens

- **Rule readout (H4):** at the query-item and decision positions, lens scores for the cued attribute's words ("mammal", "bird") vs the uncued attribute's ("wild", "pet").
- **J-space geometry (H5):** compute the Exp 1/2 geometry metrics on (a) the J-space component of h (a sparse non-negative combination of ≤ 25 lens vectors), (b) the sparse codes themselves, and (c) the span of the attribute-word concept vectors. Compare against full hidden states and dimension-matched baselines.
- **Stretch (T-044):** steer or patch with lens concept vectors to flip the grouping (a causal test).

### 4.5 Experiment 4: stretch goals

- **Ambiguous demonstrations** (A and B confounded in the demos): which grouping does the model default to, and does neutral geometry predict that preference? (T-050)
- **Scale:** repeat the core analyses across model sizes (T-051).

### 4.6 Prompt format & conditions

Candidate templates (finalized in T-016; `src/icl_geometry/tasks/templates.py`): `arrow` (below) and `io` (`Input: dog\nOutput: foo`), which puts a space before each item and doubles the share of single-token items (D-008). Default `arrow` template:

```
dog -> foo
chicken -> bar
wolf -> foo
eagle -> bar
duck ->
```

The label is scored at the final position (the **decision position**) as log p(" foo") vs log p(" bar").

| Factor | Levels | Notes |
|---|---|---|
| Experiment | 1 (category pairs), 2 (crossed attributes) | |
| Cued attribute (Exp 2) | A, B | |
| Shots m | Exp 1: 1, 2, 4, 8 per category; Exp 2: 2, 4, 8 per label (must be even, so demos balance on the uncued attribute) | Maps onto m in Eq. 1 |
| Query type (Exp 2) | congruent, incongruent | Incongruent is primary |
| Label set | ≥ 2 arbitrary single-token pairs; meaningful labels (control) | Flipped labels optional |
| Label mapping | both assignments | Counterbalances label bias |
| Demo order | ≥ 3 random orders | Recency bias |
| Condition | main; random-label control; explicit-rule control; ambiguous (Exp 4) | |
| Model | dev / main / scale | |

**Size estimate.** Exp 2 is roughly 4 domains × 2 cued attributes × 4 m × 2 mappings × ~150 prompts ≈ 10k prompts per model, plus ~30% for controls. Exp 1 is roughly (#pairs) × 4 m × ~20 prompts. Forward passes on short prompts are cheap; storage is dominated by activations (§7.4).

### 4.7 Item banks

- **Exp 1 category inventory:** THINGS (Hebart et al. 2019) high-level categories, plus finer subcategories for harder pairs.
- **Exp 2 attributes:** feature norms (McRae et al. 2005; CSLB, Devereux et al. 2014; Binder et al. 2016; THINGSplus ratings) plus curated lists. The numbers domain is generated programmatically.
- **Per-model filters:** tokenization (T-012) and zero-shot knowledge (T-013).
- **Targets:** ≥ 20, ideally 40+, usable items per category (Exp 1) or per cell (Exp 2). Geometry estimates need samples (§10).

---

## 5. Representations

- **Stream:** the residual stream after each transformer block, plus the embeddings, at every layer. For cross-model comparisons, report layers on a normalized depth of 0–100, as the J-lens paper does.
- **Positions:**
  - `query_item`: the last subtoken of the query item, i.e. what the model represents about the item given the demos;
  - `decision`: the final prompt token, where the label is predicted;
  - optionally, the demo items.
- **Neutral representations:** the item embedded in K ≥ 3 carrier templates with no demos, averaged across templates by default (D-011).
- **In-context manifolds (Exp 2):** query-item representations across prompts that share (domain, cued attribute, m, condition), grouped by the query's value on the cued attribute.
- **Circularity warning.** Late-layer decision-position states essentially encode the output. Predictive claims therefore rest on query-item representations and/or intermediate layers. The late decision-layer readout is reported only as an upper reference.
- **Outlier dimensions.** A few "massive activation" dimensions dominate Euclidean geometry in LLMs. Main results use a handling decided in advance, plus a sensitivity analysis (D-012).

---

## 6. Analyses & metrics

### 6.1 Behavioral measures

- **Per prompt:** logit margin = log p(correct label) − log p(incorrect label); correct = margin > 0; label mass = p(foo) + p(bar), which checks the model follows the format.
- **Per task:** error rate (on incongruent queries for Exp 2), both raw and contextually calibrated (Zhao et al. 2021).

### 6.2 Geometric theory (Sorscher et al. 2022, Eq. 1; verified against the paper's full text)

```
SNR_a = ½ [ ‖Δx₀‖² + (R_b²/R_a² − 1)/m ]  /  √( D_a⁻¹/m + ‖Δx₀·U_b‖²/m + ‖Δx₀·U_a‖² )
ε_a   = H(SNR_a),   H(x) = ∫ₓ^∞ e^(−t²/2)/√(2π) dt   (Gaussian tail function)
```

The terms are signal ‖Δx₀‖² = ‖x₀ᵃ − x₀ᵇ‖²/R_a²; bias (R_b²/R_a² − 1)/m; dimension D_a (participation ratio); and signal–noise overlaps, with U_a = [u₁R₁, …, u_N R_N]/R_a². Estimation details are in `docs/background/sorscher_theory.md`. Validate against the official code (github.com/bsorsch/geometry-fewshot-learning) and against Monte Carlo prototype learning.

Also implement Eq. 2, the random-projection prediction SNR(M) = SNR∞ / (1 + D∞/M). Exp 3 needs it for its dimension-matched baselines.

### 6.3 Empirical prototype learner

Monte Carlo m-shot nearest-prototype classification on the same representations (paper §L). Theory vs prototype learner checks the math; prototype learner vs LLM is the scientific comparison.

### 6.4 Shift metrics (Exp 2)

- **ΔSNR_cued** = SNR(cued partition | in-context reps) − SNR(cued partition | control reps), where the control is either the neutral reps or the random-label-context reps.
- **Selection index** = decodability(cued) − decodability(uncued), from a cross-validated linear probe, with control tasks following Hewitt & Liang (2019).
- **RSA:** partial correlations of the item dissimilarity matrices with attribute-A vs attribute-B model matrices (Kriegeskorte et al. 2008).
- **CCGP** (cross-condition generalization; Bernardi et al. 2020) for the cued attribute across the two values of the uncued attribute.
- **Per-prompt shift score:** projection of (in-context − neutral) query rep onto the cued-attribute axis. The axis is fit on training prompts only.

### 6.5 Distinguishing H2a vs H2b

| Observation | H2a (shift) | H2b (readout) |
|---|---|---|
| Uncued-attribute decodability at `query_item`, in context vs control | ↓ | ≈ |
| Cued-attribute SNR at `query_item` | ↑ | ≈ |
| Cued-attribute information at `decision` (middle layers) | ↑ | ↑ |
| Shift score predicts accuracy (H3) | yes | weak or none at `query_item`; possibly at `decision` |

### 6.6 Prediction & model comparison

- **Task level:** Spearman ρ between each predictor and LLM error across tasks.
- **Prompt level:** held-out R² for logit margin and AUROC for correctness; mixed-effects models with random intercepts for query item and demo set.
- **Predictor families:** neutral SNR (Exp 1); in-context SNR; shift score; J-lens rule readout; J-space SNR; baselines (word frequency, signal-only centroid distance, layer-0 geometry, label bias); and the circular upper reference (late decision readout).

### 6.7 Jacobian-lens analyses

- **Lens:** J_ℓ = E[∂h_final/∂h_ℓ]; readout lens(h) = softmax(W_U · norm(J_ℓ h)); concept vector v_t = J_ℓᵀ W_U[t].
- **Rule-readout score** = mean lens score(cued words) − mean lens score(uncued words). Positive control: explicit-question prompts. Negative control: random-label prompts.
- **J-space representations:** a sparse non-negative decomposition (k ≤ 25) onto lens vectors, and the span of the attribute-word vectors.
- **Dimension-matched baselines:** random subspaces (compare with Eq. 2), PCA, supervised probe directions (as an upper reference), logit lens (J = I), and optionally the tuned lens.
- **The key tension.** By Eq. 1, projecting to fewer dimensions lowers D and can lower SNR. So "J-space predicts better" is not automatic, which is what makes H5 a real test.

### 6.8 Statistics & held-out discipline

- **Splits:** items (and, for Exp 1, categories) are split into train/validation/test **before analysis** (D-009). Layer selection and hyperparameters (probe regularization, k) are chosen on validation data; headline numbers come from test data.
- **Non-independence:** prompts share items, so use item-level cross-validation, cluster bootstrap over items, and mixed-effects models.
- Report **full layer profiles**, not just the best layer.
- Fix and log all seeds. Every reported number must trace back to a `results/<run_id>/` directory.

### 6.9 Primary metrics (lightweight pre-registration; lock before T-024 runs)

1. **Exp 1:** held-out Spearman ρ between theory-predicted error (neutral reps, layer selected on validation) and LLM error across tasks (pair × direction × m).
2. **Exp 2:** ΔSNR_cued at `query_item` vs the random-label control, by layer (H2); held-out AUROC of the per-prompt shift score for incongruent correctness (H3).
3. **Exp 3:** held-out AUROC/R² of J-space predictors vs full hidden states vs dimension-matched random subspaces (H5); rule-readout AUROC (H4).

Any change made after seeing data must be logged in `docs/DECISIONS.md` with its rationale and reported as exploratory.

---

## 7. Engineering architecture

### 7.1 Repository layout (target)

```
.
├── CLAUDE.md / AGENTS.md / README.md
├── docs/            PLAN.md · DECISIONS.md · COMPUTE.md (T-003) · course/ · background/
├── tickets/         README.md · BOARD.md (generated) · _TEMPLATE.md · T-NNN-*.md
├── scripts/         tickets.py · smoke_test.py · make_prompts.py · run_*.py · make_figures.py
├── configs/         domains.yaml · prompts/*.yaml · runs/*.yaml
├── src/icl_geometry/
│   ├── tasks/       item banks, filters, prompt generation        (T-010–T-014)
│   ├── models/      loading, hooks, activation extraction          (T-003, T-020, T-030)
│   ├── behavior/    label scoring, calibration                     (T-015)
│   ├── geometry/    sorscher.py, prototype.py, probes, rsa, ccgp   (T-022, T-023, T-031)
│   ├── lens/        J-lens wrapper, readouts, J-space              (T-040–T-042)
│   ├── analysis/    prediction models, statistics                  (T-024, T-033, T-043)
│   └── io.py        prompts / behavior / activation / run-metadata I/O (T-005)
├── data/items/      curated item banks (small CSVs, committed)
├── data/cache/      activations, lens files (gitignored)
├── results/         one directory per run (gitignored except REPORT.md summaries)
├── figures/         final figures (generated by scripts/make_figures.py)
├── notebooks/       exploration only, never the source of truth
└── tests/
```

### 7.2 Data flow

```
data/items/*.csv ──T-012/T-013 filters──► valid items ──T-014──► prompts.jsonl
prompts.jsonl ──T-015──► behavior.parquet
prompts.jsonl ──T-020/T-030──► activations (fp16 memmap)
activations ──T-022/T-031──► geometry metrics ──T-024/T-033──► reports & figures
activations ──T-040–T-042 (J-lens)──► lens readouts / J-space features ──T-041/T-043──► reports & figures
```

### 7.3 Data formats

- **Prompt record (JSONL):** `prompt_id, experiment, domain, category_pair | (cued_attr, uncued_attr), m, label_set, label_map, demos[{item,label}], query_item, query_attrs, congruent, condition, correct_label, incorrect_label, template_id, seed, split, text`.
- **Behavior (parquet):** `prompt_id, model, logp_correct, logp_incorrect, margin, correct, label_mass, margin_calibrated`.
- **Activations:** `data/cache/<model>/<prompt_set_hash>/<position>.npy`, an fp16 memmap of shape `[n_prompts, n_layers+1, d_model]`, plus `index.parquet` mapping row → prompt_id.
- **Run directory:** `results/<YYYYMMDD-HHMM>_<exp>_<model>_<desc>/` with `config.yaml` (resolved), `metadata.json` (git commit, dirty flag, package versions, GPU, timings), outputs, and `REPORT.md` where relevant.

### 7.4 Compute & storage estimates

- **Activations:** bytes ≈ n_prompts × n_positions × (n_layers+1) × d_model × 2 (fp16). Example: 10k prompts × 2 positions × 37 × 2560 × 2 B ≈ 3.8 GB for a 4B-class model (verify the dimensions in T-003). Subsample layers if needed.
- **Lens application** is one d × d matmul per layer, which is cheap. **Fitting our own lens** (T-045) needs backward passes over ~100–1000 sequences of 128 tokens.

### 7.5 Tooling

Python ≥ 3.11 with `uv`; PyTorch and HF `transformers`, plus `jlens` (pinned commit); numpy, scipy, pandas, pyarrow, scikit-learn, statsmodels, matplotlib. Tests use `pytest`; linting uses `ruff`. `scripts/tickets.py` stays stdlib-only.

---

## 8. Phases, milestones & tickets

| Phase | Goal | Tickets | Milestone |
|---|---|---|---|
| 0 Setup | Environment, compute, J-lens spike, conventions, course logistics | T-001–T-007 | **M0:** environment works and lens verified |
| 1 Tasks & behavior | Item banks, filters, prompts, scoring, pilot | T-010–T-016 | **M1:** pilot shows usable dynamic range |
| 2 Neutral geometry | Theory implementation and Exp 1 | T-020–T-024 | **M2: Exp 1 result (MVP)** |
| 3 In-context geometry | Exp 2: shift, H2, H3 | T-030–T-034 | **M3:** H2/H3 verdict |
| 4 Jacobian lens | Exp 3: H4, H5 | T-040–T-045 | **M4:** J-lens comparison |
| 5 Stretch | Ambiguous demos, scaling | T-050–T-051 | — |
| 6 Write-up | Figures, report, presentation, reproducibility | T-060–T-063 | **M5:** submission |

**Critical path:** T-002 → T-003 → T-010 → T-011 → T-012 → T-014 → T-015 → T-013 → T-016 → T-020 → T-021 → T-024 → T-061.

**Can start immediately, in parallel:** T-001 (human: course logistics), T-006 (reading), T-010 (domain design), T-002 → T-022/T-023 (theory code needs only the environment). Start T-004 (J-lens spike) as soon as T-003 is done, because it decides the model choice.

**Scope rule:** P2 tickets start only after M2. If time runs short, cut in this order: T-050, T-051, T-044, T-045, part of T-034; then reduce T-042/T-043 to the attribute-span subspace only.

---

## 9. Timeline

**TBD:** depends on the deadlines collected in T-001. Suggested share of the schedule:

| Phase | Share of schedule |
|---|---|
| 0–1 Setup, tasks, behavior | ~25% |
| 2 Neutral geometry (MVP) | ~20%, so the MVP lands by the ~45–50% mark |
| 3 In-context geometry | ~20% |
| 4 Jacobian lens | ~15% |
| 6 Write-up (overlaps with analysis) | ~20% |

---

## 10. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation | Tickets |
|---|---|---|---|---|
| Small models fail at ICL on harder groupings (floor effect) | M | H | Pilot early; use 4B/9B; adjust difficulty; more shots | T-016 |
| Tasks at ceiling, leaving no variance to predict | M | H | Harder pairs, arbitrary labels, fewer shots | T-016 |
| Pre-fitted lens mismatched (wrong checkpoint; J++ ≠ J-lens) | M | M | Verify early; fit our own lens on a small model | T-004, T-045 |
| J-space weak or incoherent in small models | M | M | Report it as a finding; compare with the logit lens; try a larger model | T-041, T-051 |
| Too few items for stable geometry (biased D, R) | M | M | Item-count targets; multiple templates; bootstrap; Eq. 2 corrections | T-011, T-021 |
| Outlier dimensions dominate distances | H | M | Handling decided in advance, plus sensitivity analysis | T-021 |
| Circular predictors at the decision position | H | H | Primary claims at `query_item` / middle layers; late readout as upper reference only | T-033 |
| Double dipping in layer selection | M | H | Validation/test splits fixed up front | T-014, T-024 |
| Label or recency bias confounds accuracy and asymmetry | H | M | Counterbalance mapping and order; calibration | T-014, T-015 |
| Knowledge failures masquerade as ICL failures | M | M | Zero-shot knowledge filter | T-013 |
| Compute or storage limits | M | M | Develop on 0.8B; subsample layers; fp16 memmaps | T-003, T-020 |
| `jlens` (unmaintained) breaks with newer `transformers` | M | M | Pin versions; keep a minimal reimplementation of the readout | T-004, T-040 |
| Scope creep | H | H | MVP first; P2 only after M2; cut list in §8 | — |

---

## 11. Open questions

- Deadlines, deliverable format, team, compute (T-001).
- Which checkpoint (base vs post-trained) each pre-fitted lens matches (T-004).
- J++ vs the original J-lens: does the difference matter for our claims? (T-004, T-045)
- Neutral-template aggregation and outlier-dimension handling (T-021).
- How many Exp 2 domains are enough (T-016).
- Whether multi-token items are acceptable (T-012).

---

## 12. References

- Sorscher, Ganguli & Sompolinsky (2022). Neural representational geometry underlies few-shot concept learning. *PNAS* 119(43): e2200800119. https://www.pnas.org/doi/10.1073/pnas.2200800119 · code: https://github.com/bsorsch/geometry-fewshot-learning
- Gurnee, Sofroniew, Pearce, … Batson & Lindsey (2026). Verbalizable representations form a global workspace in language models. arXiv:2607.15495 · code: https://github.com/anthropics/jacobian-lens · lenses: https://huggingface.co/neuronpedia/jacobian-lens
- Mante, Sussillo, Shenoy & Newsome (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature*.
- Flesch, Juechems, Dumbalska, Saxe & Summerfield (2022). Orthogonal representations for robust context-dependent task performance in brains and neural networks. *Neuron*.
- Bernardi, Benna, Rigotti, Munuera, Fusi & Salzman (2020). The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell*.
- Freedman, Riesenhuber, Poggio & Miller (2001). Categorical representation of visual stimuli in the primate prefrontal cortex. *Science*.
- Kriegeskorte, Mur & Bandettini (2008). Representational similarity analysis. *Front. Syst. Neurosci.*
- Chung, Lee & Sompolinsky (2018). Classification and geometry of general perceptual manifolds. *PRX*.
- Hewitt & Liang (2019). Designing and interpreting probes with control tasks. arXiv:1909.03368.
- nostalgebraist (2020). Interpreting GPT: the logit lens. · Belrose et al. (2023). Eliciting latent predictions with the tuned lens. arXiv:2303.08112.
- Zhao et al. (2021). Calibrate before use. *ICML*. · Min et al. (2022). Rethinking the role of demonstrations. · Wei et al. (2023). Larger language models do in-context learning differently. arXiv:2303.03846.
- Hendel, Geva & Globerson (2023). In-context learning creates task vectors. · Todd et al. (2024). Function vectors in large language models. *ICLR*.
- Hebart et al. (2019). THINGS. *PLoS ONE*. · McRae et al. (2005) feature norms. · Devereux et al. (2014) CSLB norms. · Binder et al. (2016) componential semantic representation.
