# Does representational geometry predict in-context generalization in a language model?

_Neuro 240 / Phys 240, Biological and Artificial Intelligence, Fall 2026 · Final project (Course Project 5) · Yashwanth Ravipati_

> **DRAFT SKELETON (agent-drafted, 2026-10-27).** The student owns the final text. The Introduction and Methods are first drafts written from `docs/PLAN.md`, `docs/DECISIONS.md`, `docs/background/` and the code in `src/icl_geometry/`. **Results contain no numbers**: every result is a `RESULT PENDING` box naming the run directory and figure it will come from. Unresolved choices are marked `[TBD: …]`. Format and length depend on T-001: `[TBD: T-001 — report format, page/word limit, citation style, deadline]`.

---

## Abstract

> **PLACEHOLDER.** Write last, after the Exp 1 (H1) result is in. Suggested shape (~200 words): the question (can a neuroscience theory of few-shot learning from representational geometry predict LLM in-context learning?); the system (Qwen3.5-2B-Base); the three experiments; one sentence per headline result, nulls stated plainly; and one sentence on what the results say back to neuroscience.

---

## 1. Introduction

**Few-shot concept learning and representational geometry.** Primates can learn a new category from a handful of examples. Sorscher, Ganguli & Sompolinsky (2022) showed that a single downstream neuron using a simple *prototype* rule (average the m examples of each concept, then classify a new example by the nearest prototype) can reach high few-shot accuracy on top of good representations. They also derived an analytical theory that predicts the error of this learner from a few measurable properties of the *concept manifolds* (the point clouds of responses to a concept's examples):
- the distance between manifold centroids (signal);
- the manifolds' relative sizes (bias);
- their dimensionality (participation ratio);
- the overlap between the signal direction and the manifolds' noise directions.

The theory matched few-shot learning on deep vision networks and on recordings from macaque IT cortex, which makes it a quantitative bridge between neural geometry and behavior.

**Flexible regrouping of the same stimuli.** Categories are not fixed. The same stimuli can be grouped differently depending on the task, and prefrontal neurons come to reflect learned category boundaries (Freedman, Riesenhuber, Poggio & Miller, 2001). There are two accounts of how context selects the relevant grouping:
- **Readout selection.** In monkey PFC during context-dependent decisions, the task-irrelevant stimulus dimension is *still represented* but is not integrated into the choice. Selection happens in the recurrent dynamics and readout (Mante, Sussillo, Shenoy & Newsome, 2013).
- **Representational compression.** Humans, and networks trained in the "rich" regime, instead form orthogonal task representations and *compress* the irrelevant dimension (Flesch, Juechems, Dumbalska, Saxe & Summerfield, 2022).

A third geometric notion, the *abstract format* of a variable (Bernardi et al., 2020), asks whether a decoder trained in one condition generalizes to others (cross-condition generalization performance, CCGP).

**A global workspace.** Global workspace theories propose that a limited-capacity workspace broadcasts selected content that is then reportable (Baars, 1988; Dehaene, Kerszberg & Changeux, 1998). Gurnee et al. (2026) recently introduced the *Jacobian lens* (J-lens) for language models. It transports an internal activation to the final layer with the average input–output Jacobian and decodes it with the model's own unembedding. They argue that the sparse, "verbalizable" component of the residual stream it reveals, the *J-space*, behaves like a global workspace. Of particular relevance here, they report that implicit properties (part of speech, tense) enter the J-space only when the task asks for them.

**LLM in-context learning as a test bed.** Large language models learn categorizations from demonstrations in the prompt with no weight updates (`dog -> foo, cat -> foo, oak -> bar, fern -> bar, bat ->`). Unlike a brain, we can record every "neuron" at every layer, intervene freely, and present thousands of controlled trials. We use this to ask three questions:
1. **Exp 1 (H1).** Does the Sorscher theory, applied to an LLM's context-free ("neutral") item representations, predict its in-context few-shot error across category pairs and numbers of shots? This is a direct test of a neuroscience theory on a system it was not designed for.
2. **Exp 2 (H2a/H2b, H3).** When the demonstrations regroup the same items (mammal/bird vs domestic/wild), do the query-item representations reorganize toward the demonstrated grouping and compress the other one (Flesch-like, H2a)? Or do both attributes stay represented, with selection only near the readout (Mante-like, H2b)? Does the size of the reorganization predict generalization to unseen items (H3)?
3. **Exp 3 (H4, H5).** Is the demonstrated rule "reportable" in the model, i.e. do the cued attribute's words rise in the J-lens readout (H4)? And does geometry in the J-space predict generalization better than full hidden states, against dimension-matched baselines (H5)?

Throughout, null results are treated as findings. Where the theory fails, its failure points to how in-context learning differs from a prototype learner.

`[TBD: tighten to the T-001 length limit; add 1–2 sentences on prior work on ICL mechanisms (task vectors: Hendel, Geva & Globerson 2023; function vectors: Todd et al. 2024; the role of labels in demonstrations: Min et al. 2022; Wei et al. 2023) if space allows.]`

---

## 2. Methods

### 2.1 Models

- **Main model:** `Qwen/Qwen3.5-2B-Base`.
- **Development model:** `Qwen/Qwen3.5-0.8B`, a post-trained checkpoint, used only for building and smoke-testing the pipeline.
- **Architecture.** Both are 24-block models (residual width d = 2048 and 1024 respectively) with a hybrid architecture: three Gated-DeltaNet linear-attention blocks per full-attention block. The J-lens paper's open-model demonstrations used standard transformers, so this is a point of departure worth stating.
- **Prompting.** Raw text completion, with no chat template.
- **Choice of model (D-003, revised).** We chose the main model because it is a base checkpoint, which suits raw few-shot completion, and because a plain Jacobian lens was fitted on exactly this checkpoint (§2.8).
- **Optional scaling (T-051).** Qwen3.5-4B and Qwen3.5-9B-Base, which also have lenses. `[TBD: T-051 — include only if run.]`
- **Inference.** fp32 (CPU) or bf16 (Apple MPS/GPU). `[TBD: report the hardware and dtype actually used for the headline runs, from each run's metadata.json.]`

### 2.2 Residual-stream representations

- **Where states are recorded.** We record the residual stream with forward hooks on each decoder block. State index 0 is the token embeddings, and index l + 1 is the output of block l, giving 25 states. This is the same convention the `jlens` package uses (verified identical to `jlens.ActivationRecorder`). We do not use Hugging Face's last `hidden_states` entry, because it already has the final norm applied.
- **Normalized depth.** Layers are reported on a 0–100 scale (0 = embeddings), as in the J-lens paper.
- **Two token positions (D-006):**
  - `query_item`: the last subtoken of the query item.
  - `decision`: the final prompt token, where the label is predicted.
- **Multi-token items (D-008).** Multi-token items are kept and represented by their last subtoken. The number of subtokens is stored as a covariate, and a single-token-subset sensitivity analysis is planned.
- **Circularity.** Late-layer `decision` states essentially encode the output. Primary predictive claims therefore rest on `query_item` states and intermediate layers. The last-layer `decision` readout is reported only as a labelled upper reference.

**Neutral representations (D-011).**
- **Carriers.** Each item is embedded in four carrier texts with no demonstrations: "The word is {item}", "Here is one example: {item}", "Item: {item}" and "Next line:\n{item} ->".
- **Readout.** We read the state at the item's last subtoken in each carrier. The neutral representation is the mean over carriers.
- **Design choices.** The item is never at position 0, because Qwen has no BOS token and position 0 acts as an attention sink. The carriers mix the two tokenizations of an item: after a space, and at the start of a line, as in the `arrow` template.
- `[TBD: D-011 sign-off; the carriers-as-extra-samples sensitivity analysis.]`

**Outlier dimensions (D-012).**
- **Primary analysis:** raw residual states, with no standardization or whitening, because the theory's quantities are defined in the raw Euclidean space.
- **Sensitivity analyses:** (i) drop the top-5 dimensions by mean |activation|; (ii) per-dimension standardization.
- `[TBD: D-012 — re-check the top-1 variance share on 2B-Base before locking.]`

### 2.3 Tasks and item banks

**Experiment 1: category pairs (D-007).**
- **Items:** 31 curated categories under 7 superordinates (animal, plant, food, artifact, body, person, nature), 970 items in total. Items are lowercase singular common nouns, each belonging to exactly one category.
- **Pairs:** all 465 unordered category pairs. The 95 pairs within a superordinate are the semantically related, harder pairs that the course brief asks for. Relatedness is defined as a shared superordinate.
- **Confidence:** items are curated with a confidence label. Main analyses use only `high`-confidence items.
- **Task:** a *task* is (a, b, m): test examples of category a against category b, with m demonstrations per category. Each unordered pair gives two directed tasks.
- `[TBD: D-007 sign-off and the T-011 item spot-check. Note in the write-up that the source is project curation, not THINGS or McRae norms.]`

**Experiment 2: crossed 2×2 domains (D-007).** Items vary on two binary attributes, and only the demonstrations say which one defines the labels.

| Domain | Attribute A | Attribute B | Note |
|---|---|---|---|
| Animals | mammal / bird | domestic / wild | Domestication is the least objective attribute; few unambiguous domestic birds |
| Countries | Europe / Asia | coastal / landlocked | Objective; transcontinental countries excluded; small Asia-landlocked cell |
| Numbers | even / odd | < 50 / > 50 | Two-digit numbers; knowledge-free control |
| Words | animal / food | first letter b / p | Semantic vs surface attribute |

- **Foods (fruit/vegetable × red/green) was dropped**, because colour is not objective for most produce.
- **Numbers confound (D-008).** Numbers always tokenize as two digits, so the `query_item` state sits on the units digit. That digit carries parity directly but not magnitude, which confounds `query_item` geometry in the numbers domain.
- **Congruent vs incongruent queries.** A query is *congruent* if both rules give the same label, and *incongruent* if they disagree. Incongruent queries are the diagnostic condition.
- `[TBD: D-007 sign-off.]`

**Tokenization and knowledge filters.**
- **Tokenization.** Labels and J-lens attribute words must be single tokens with a leading space under the model's tokenizer. The 0.8B and 2B-Base models share an identical vocabulary.
- **Replacements.** The label pair dax/wug became dax/mip, and the attribute words " mammal" and " landlocked" were replaced by single-token alternatives (" mammals", " fur"; " inland", " land").
- **Knowledge filter (T-013).** A zero-shot forced-choice knowledge check is meant to remove items whose category or attribute the model does not know. Without it, knowledge failures could masquerade as in-context learning failures.
- `[TBD: T-013 — state whether the filter was applied and how many items it removed.]`

### 2.4 Prompts and controls

**Templates** (`src/icl_geometry/tasks/templates.py`). Labels carry their leading space, and the prompt ends right before the label.

    arrow:                       io:
    dog -> foo                   Input: dog
    chicken -> bar               Output: foo
    wolf -> foo
    eagle -> bar                 Input: chicken
    duck ->                      Output: bar
                                 ...
                                 Input: duck
                                 Output:

`[TBD: D-010 — the final template. At the start of a line (arrow), items are single tokens only ~22% of the time, vs ~53% after a space (io), so tokenization is one criterion.]`

**Design factors.**

| Factor | Levels |
|---|---|
| Shots m | Exp 1: 1, 2, 4, 8 per category. Exp 2: 2, 4, 8 per label; m must be even so demos balance on the uncued attribute. `[TBD: D-010 m grid]` |
| Label set | foo/bar (default), zip/zap, dax/mip, plus a meaningful-label control (the attribute value names). Current main configs: Exp 1 foo/bar; Exp 2 foo/bar and zip/zap. `[TBD: D-010]` |
| Label mapping | Both assignments of labels to groups (counterbalances label bias) |
| Demo order | Several random orders per demo set (current configs: 1 for Exp 1, 2 for Exp 2) `[TBD: D-010]` |
| Condition (Exp 2) | `main`; `random_label` (same demo items, labels permuted so no attribute predicts them; the negative control); `explicit_rule` (main demos preceded by a one-line stated rule; positive control); `ambiguous` (stretch, Exp 4) |

**Exp 2 demonstration balance.** Exp 2 demonstrations are balanced on the uncued attribute, so only the cued attribute explains the labels.

**Splits (D-009).**
- **Procedure.** Items are split once into **validation (40%) and test (60%)**, stratified by category (Exp 1) or by crossed cell (Exp 2). Within each stratum, items are sorted on a salted SHA-256 hash of the item string, so the split is stable across machines and independent of row order.
- **No shared items.** Every prompt set draws its demonstrations and queries from a single split, so validation and test prompts share no items.
- **Exp 1 caveat.** Exp 1 shares *categories* (not items) across splits.
- `[TBD: D-009 confirmation.]`

### 2.5 Behavioral measures

- **One forward pass per prompt.** It yields both the label log-probabilities at the `decision` position and the cached residual states (right padding with per-row indexing; pad tokens cannot affect earlier positions in a causal model).
- **Margin and correctness.** Margin = log p(correct label) − log p(incorrect label). A trial is correct if the margin > 0.
- **Label mass.** p(label 1) + p(label 2) checks that the model follows the format.
- **Contextual calibration (Zhao et al., 2021).** The same demonstrations are scored with the content-free query "N/A". Its label log-odds are subtracted from the margin, giving `margin_calibrated` and `correct_calibrated`.
- **Task-level error.**
  - **Exp 1:** for task (a, b, m), the error rate on prompts whose query comes from category a.
  - **Exp 2:** the primary measure is accuracy on incongruent queries.

### 2.6 The geometric theory of few-shot learning

**Manifold statistics** (`src/icl_geometry/geometry/sorscher.py`, ported from the official code's `geometry()` function). For a category with P item representations (a P × N matrix):
- **Centroid** x₀: the mean representation.
- **Radii and axes.** R and U are the singular values and right singular vectors of the centered sample matrix (a thin SVD; the N × N covariance is never formed). The covariance eigenvalues are Rᵢ²/P.
- **Total variance.** tr(C) = Σᵢ Rᵢ²/P.
- **Dimension.** D = (Σ R²)² / Σ R⁴, the participation ratio, which is at most P − 1.

**Pair terms.** For test examples of a against b, with Δx₀ = x₀ᵃ − x₀ᵇ and unit direction Δx̂₀:
- **Signal:** ‖Δx₀‖²/tr(C_a).
- **Bias:** (tr C_b / tr C_a − 1)/m.
- **Signal–noise overlaps:** csa = Σᵢ (uᵢᵃ·Δx̂₀)² Rᵢᵃ² / Σ Rᵃ² and csb = Σᵢ (uᵢᵇ·Δx̂₀)² Rᵢᵇ² / Σ Rᵃ², each multiplied by the signal.
- **Subspace–subspace overlap:** ss = Σᵢⱼ (uᵢᵃ·uⱼᵇ)² Rᵢᵃ² Rⱼᵇ² / (Σ Rᵃ²)². It appears in the official code but not in the short form of Eq. 1.

**SNR and predicted error:**

    SNR_a = ½ (signal + bias) / sqrt( 1/(D_a m) + overlap_a + overlap_b/m + ss/m
                                      [+ 1/(2 D_a m²) + (tr C_b/tr C_a)/(2 D_b m²)] )
    ε_a   = H(SNR_a),   H = Gaussian tail function

- **Primary variant (D-013).** The primary prediction is the *full* expression, including the bracketed 1/m² terms from SI §3C. The dominant-terms version is reported as a sensitivity check.
- **Why the full variant.** On synthetic manifolds it is closer to Monte Carlo at small m, and our m grid starts at 1.
- **Validation (T-022).** The implementation matches a verbatim port of the official code to 10⁻⁸. It was also validated against Monte Carlo prototype learning on synthetic Gaussian manifolds: agreement is close for high-D manifolds, while for low-D (power-law) manifolds the theory overestimates error but preserves rank order. `[TBD: cite run results/20261026-0111_t022_synthetic_theory-vs-mc/ and figure theory_vs_prototype_synthetic, or move to a supplement.]`
- **Random projections (Eq. 2).** SNR(M) = SNR∞/(1 + D∞/M). We checked numerically that it holds only for m → ∞ and M ≪ d, so it is reported only as an asymptotic reference. Empirical random subspaces serve as the null in Exp 3.

**Empirical learners** (`geometry/prototype.py`, T-023).
- **Monte Carlo procedure.** On the same representations, each draw samples m training items per class without replacement, and all remaining items are test items, as with held-out queries. Ties count as half an error.
- **Two rules:** `prototype` (nearest class mean, the rule the theory describes) and `exemplar` (1-nearest-neighbour). Comparing them with the LLM asks which learning rule in-context learning most resembles.
- **What the comparisons test.** Theory vs simulated prototype learner checks the approximation; simulated learner vs LLM is the scientific comparison.

**Mapping onto in-context learning.**

| Sorscher | This project |
|---|---|
| Neurons | Residual-stream dimensions at one layer |
| Concept and its examples | Category, and its items' neutral representations |
| m training examples | m demonstrations per category |
| Test example | Held-out query item |

### 2.7 Experiment 2: shift metrics

**In-context manifolds.** These are the `query_item` (or `decision`) states across prompts that share (domain, cued attribute, m, condition), partitioned by the query's value on the cued or uncued attribute. Every in-context quantity is compared with the **random-label control**. That control has the same demo items, prompt length and label tokens; only the cued structure differs. The comparison isolates the effect of the demonstrated grouping from generic context effects (`src/icl_geometry/analysis/exp2.py`).

**Metrics** (`src/icl_geometry/geometry/shift.py`), computed per (domain, cued attribute, m) cell and layer:
- **ΔSNR_cued / ΔSNR_uncued.** The SNR of the cued (or uncued) partition, defined as the mean of Eq. 1 (full) over both directions, in the `main` condition minus the `random_label` condition.
- **Decoding accuracy and selection index.**
  - Decoding accuracy is cross-validated, with folds split by query item because the same item appears in many prompts.
  - The default decoder is nearest-class-mean (a prototype decoder): with tens of items in d ≥ 1024 dimensions, a logistic probe overfits item identity. A strongly L2-regularized logistic probe is the alternative.
  - Selection index = accuracy(cued) − accuracy(uncued).
  - A Hewitt & Liang (2019) control task (a random label per item, held constant across its prompts) calibrates selectivity.
- **Δ decoding accuracy.** Main − random-label, for each attribute.
- **CCGP (Bernardi et al., 2020).** Train the cued decoder on one value of the uncued attribute and test on the other, averaged over both directions.
- **RSA (Kriegeskorte et al., 2008).** Partial Spearman correlation of the item dissimilarity matrix (correlation distance) with each attribute's model matrix, controlling for the other attribute.
- **Per-prompt shift score (H3).**
  - For each (domain, cued attribute), a cued axis is fitted as the difference of class means of `main` in-context `query_item` states, using validation prompts only.
  - The shift score is the signed projection of (in-context − neutral) state on that axis. Positive means the state moved toward the item's own cued class. Units are half the class-mean separation.

**How H2a and H2b are distinguished (PLAN §6.5).**

| Observation | H2a (shift / compression) | H2b (readout selection) |
|---|---|---|
| Uncued decodability at `query_item`, main vs control | ↓ | ≈ |
| Cued SNR at `query_item` | ↑ | ≈ |
| Cued information at `decision` (middle layers) | ↑ | ↑ |
| Shift score predicts accuracy (H3) | yes | weak or none at `query_item` |

### 2.8 Experiment 3: the Jacobian lens

**Lenses.**
- **Main model:** the plain J-lens fitted on Qwen3.5-2B-Base, from Hugging Face `neuronpedia/jacobian-lens`, file `qwen3.5-2b-pt/jlens/Salesforce-wikitext/…` (WikiText-103, Anthropic's `jlens` code, early-stopped at 335 prompts).
- **Development model:** the corresponding 0.8B lens.
- **Package version.** `jlens` is pinned at commit 581d398. We use it only for the lens file format and reimplemented the readout over cached states (`src/icl_geometry/lens/`).
- **J++.** The J++ lenses are a different variant and are not used for primary claims.

**Readout.** lens_l(h) = lm_head(final_norm(J_l h)). Lens layer l transports the output of block l, which is cached state index l + 1. The **logit lens** (J = I) serves as the baseline.

**Concept vectors.** The final RMSNorm has an effective per-dimension gain g (for Qwen3.5 the gain is 1 + w, measured numerically as norm(1)). The concept vector for token t is v_t = J_lᵀ(g ⊙ W_U[t]). Within one position, ranking tokens by ⟨v_t, h⟩ is exactly the lens ranking.

**Rule readout (H4)** (`lens/readout.py`).
- **Score.** For each prompt, position and lens layer, each attribute word's percentile rank in the full-vocabulary lens distribution (1 = top). Ranks are comparable across positions and layers; raw logits are not.
- **Rule readout** = mean rank of the cued attribute's words − mean rank of the uncued attribute's words.
- **Comparisons:** main vs `random_label` (negative control) vs `explicit_rule` (positive control), and correct vs incorrect trials.

**J-space (H5)** (`lens/jspace.py`).
- **Decomposition.** Each residual is decomposed by greedy non-negative gradient pursuit into at most k = 25 unit-normalized concept vectors.
- **Dictionary.** The dictionary is a concept vocabulary: space-prefixed lowercase words of at least 3 letters, plus all attribute-word tokens. The default size is 8,000 tokens, because the full ~248k-token vocabulary is too large per layer.
- **Representations compared**, at three middle lens layers by default:
  - `full`: the full hidden state;
  - `jspace`: its J-space reconstruction;
  - `attr_span`: coordinates in the span of the attribute words' J-lens vectors (rank r);
  - `logit_span`: the same span under the logit lens;
  - `pca_r`: PCA on validation main-condition states, with r components;
  - `random_r`: r-dimensional random orthonormal subspaces, over 5 seeds.
- **Evaluation.** For each representation we compute the shift score and ΔSNR_cued, along with the J-space variance explained and the J-space participation ratio.
- `[TBD: choose the J-space layers on validation data or fix them in advance; state which.]`

### 2.9 Statistics and held-out discipline

- **Pre-specified primary metrics** (PLAN §6.9; D-005, to be locked before T-024 runs). `[TBD: D-005 lock date.]`
  1. **Exp 1:** held-out Spearman ρ between theory-predicted error (neutral representations, layer selected on validation) and LLM error, across tasks (pair × direction × m).
  2. **Exp 2:** ΔSNR_cued at `query_item` vs the random-label control, by layer (H2). Held-out AUROC of the per-prompt shift score for incongruent correctness (H3).
  3. **Exp 3:** held-out AUROC of J-space predictors vs full hidden states vs dimension-matched random subspaces (H5). Rule-readout AUROC (H4).
- **Layer and hyperparameter selection** use validation data only. Headline numbers come from the test split. Full layer profiles are always shown.
- **Exp 1 inference.**
  - 95% CIs come from a bootstrap over *categories*, because tasks share categories: a resampled task (a, b) is weighted by count(a) × count(b).
  - The null distribution permutes LLM errors across tasks (1,000 permutations; one-sided p).
  - Baselines: signal-only (centroid distance) and layer-0 (embedding) geometry.
- **Exp 2/3 prompt-level prediction** (`analysis/prediction.py`).
  - A standardized logistic model is fitted on the validation split and scored on test (main condition, incongruent queries).
  - 95% CIs come from a bootstrap over query items. The incremental AUROC of the shift score over a baseline uses a paired bootstrap.
  - The H3 baseline is label bias (the content-free calibration margin) plus m.
  - A mixed-effects model (random intercept for the query item, variance component for the demo set) is available. `[TBD: say whether it is reported.]`
- **Seeds and traceability.** All seeds are fixed, and every reported number traces to a `results/<run_id>/` directory, whose resolved `config.yaml` and `metadata.json` record the seed, git commit, package versions and device.
- **Exploratory changes.** Any analysis changed after seeing data is logged in `docs/DECISIONS.md` and labelled exploratory.

---

## 3. Results

> Every subsection below is a slot. **No numbers appear here until the corresponding test-split run exists.** Pilot and smoke-test outputs on the 0.8B development model check the plumbing only and are not results.

### 3.0 Behavioral dynamic range (pilot)

Pilot on validation items only (T-016): is the task neither at floor nor at ceiling, does incongruent accuracy exceed the random-label control, and is label mass high? This section justifies D-010.

> **RESULT PENDING** — source: `results/pilot/REPORT.md` (2B-Base pilot, T-016); figures `exp2_incongruent_vs_m`, `exp2_conditions`, `exp1_learning_curves`.

### 3.1 Experiment 1: neutral geometry predicts few-shot error? (H1)

**Pre-specified test.** We compute the Spearman ρ between theory error ε_a (Eq. 1, full; neutral representations) and LLM error across test-split tasks (a, b, m), at the layer chosen on validation data. H1 is supported if ρ exceeds the 95th percentile of the permutation null. We also report the bootstrap CI over categories, the signal-only and layer-0 baselines, and the theory vs simulated prototype learner (and exemplar learner) agreement.

> **RESULT PENDING** — source: `results/<YYYYMMDD-HHMM>_exp1_qwen3.5-2b-base_analysis/` (`summary.json`, `layer_profile.csv`, `tasks.parquet`).
> - Figure slot **exp1_rho_by_layer**: ρ(theory, LLM error) by normalized depth, val and test, with the signal-only baseline and the null's 95th percentile.
> - Figure slot **exp1_scatter**: theory vs LLM error at the selected layer, coloured by m.
> - Figure slot **exp1_learning_curves**: mean error vs m for the LLM, the theory and the prototype simulation.

Prompts for the write-up (fill in only from the run):
- Does the theory track the *m*-dependence (learning curves), the *pair* dependence, or both?
- Error asymmetry ε_a ≠ ε_b: does the bias term predict it once label bias is controlled (calibrated error)?
- Related (within-superordinate) vs unrelated pairs.

### 3.2 Experiment 2: shift or selection? (H2a vs H2b)

**Pre-specified test.** ΔSNR_cued and ΔSNR_uncued at `query_item`, main vs random-label control, by layer, on test. Decoding-accuracy deltas, CCGP and RSA are reported as supporting measures, and the same profile at `decision` gives the readout side. The verdict follows the §2.7 table.

> **RESULT PENDING** — source: `results/<YYYYMMDD-HHMM>_exp2h2_qwen3.5-2b-base_h2/` (`h2_metrics.parquet`, `H2_REPORT.md`).
> - Figure slot **h2_profile**: 2 × 2 panels (rows: ΔSNR, Δ decoding accuracy; columns: `query_item`, `decision`), cued vs uncued, mean and IQR over domain × cue × m.

Also report: the behavioral prerequisite (incongruent accuracy, main vs random-label vs explicit-rule) and per-domain breakdowns. For the numbers domain, flag the units-digit parity confound.

### 3.3 Experiment 2: does shift predict generalization? (H3)

**Pre-specified test.** Test-split AUROC, on main-condition incongruent queries, of (i) the baseline (label bias + m), (ii) the shift score, and (iii) baseline + shift score, all fitted on validation data at the layer chosen on validation data. We also report the incremental AUROC of the shift score with a paired bootstrap CI. The last-layer `decision` projection is reported separately as a **circular** upper reference.

> **RESULT PENDING** — source: `results/<YYYYMMDD-HHMM>_exp2h3_qwen3.5-2b-base_h3/` (`h3_results.csv`, `summary.json`, `layer_profile_val.csv`).
> - Figure slot **h3_auroc**: test AUROC with 95% CIs per predictor set, plus the validation layer-selection profile.

### 3.4 Experiment 3: is the rule "reportable"? (H4)

**Pre-specified test.** Rule readout (rank of cued − uncued attribute words) by layer and position, compared across main, random-label and explicit-rule conditions, and the AUROC of the rule readout for incongruent correctness. The same analysis under the logit lens is the baseline.

> **RESULT PENDING** — source: `results/<YYYYMMDD-HHMM>_exp3readout_qwen3.5-2b-base_jlens/` and the matching `--logit-lens` run (`readout.parquet`, `H4_REPORT.md`).
> - Figure slot **h4_readout**: rule readout vs normalized depth by condition, `query_item` and `decision`, J-lens and logit lens.

### 3.5 Experiment 3: J-space vs full states vs matched baselines (H5)

**Pre-specified test.** Test AUROC of the shift score computed in each representation (`full`, `jspace`, `attr_span`, `logit_span`, `pca_r`, `random_r` mean ± sd). H5 is supported if J-space ≥ full and > matched random subspaces. We also report ΔSNR_cued per representation, J-space variance explained and J-space D, with Eq. 2 as the asymptotic reference only.

> **RESULT PENDING** — source: `results/<YYYYMMDD-HHMM>_exp3jspace_qwen3.5-2b-base_h5/` (`h5_auroc.csv`, `geometry.csv`, `H5_REPORT.md`).
> - Figure slot **h5_auroc**: test AUROC per representation and lens layer, with the random-subspace band.

### 3.6 Robustness and stretch results

> **RESULT PENDING** — source: the T-034 robustness run (`ROBUSTNESS.md`): label sets and mappings, demo orders, template, m, D-011/D-012 alternatives, and a second model size. Stretch work only if done: ambiguous demonstrations (T-050), scaling (T-051), causal steering (T-044). `[TBD: drop this section if none were run.]`

---

## 4. Discussion

_Scaffold. Write from the results; do not pre-commit to an interpretation._

**What transfers from neuroscience theory to LLMs, and what doesn't.**
- If H1 holds: in-context learning behaves *as if* a prototype readout sits on top of the model's concept geometry, the same account that explained IT-based few-shot learning. Compare the strength of the fit with Sorscher et al.'s DNN and IT results, qualitatively and without overclaiming.
- If H1 fails or is weak, the theory becomes diagnostic. Candidate reasons (from `docs/background/sorscher_theory.md`):
  - ICL is exemplar- or rule-like rather than prototype-like; check the exemplar learner comparison.
  - Representations change with context, which is exactly what Exp 2 measures.
  - Small P per manifold biases D downward.
  - Label and format biases add error the theory doesn't model.
  - Anisotropy.
  - No single layer is "the" representation.
- Which term carries the prediction (signal vs dimension vs overlaps)? Does a signal-only baseline do as well?

**Mante vs Flesch in a fully observed system.**
- State plainly which pattern held at `query_item` vs `decision` and at which depths. Mixed outcomes (e.g. selection near the readout but shift in middle layers) are informative, not failures.
- Caveat: *decodable ≠ used*. Unless T-044 (causal steering/patching) was done, all Exp 2/3 claims are correlational.
- Predictions back to neuroscience: what would the LLM pattern predict for PFC/IT recordings during rule-based categorization (e.g. compression of the irrelevant dimension in early vs late stages)?

**A workspace-style readout.**
- Does the demonstrated rule become "reportable" (H4) without being asked for, unlike the J-lens paper's implicit-property result? Does reportable content carry the generalization signal (H5), or is it a small (≤ ~10% of variance) side channel?
- Projecting to fewer dimensions lowers D and can lower SNR (Eq. 1), so J-space "winning" was never automatic. Interpret H5 against the dimension-matched nulls only.

**How to frame nulls.**
- State the pre-specified test, the estimate and CI, and the null. Say "no evidence for X at this power", not "X is false", and give the CI width so the reader can judge power.
- Don't search for a layer, metric or subset after the fact that "works"; anything exploratory is labelled as such.

**Limitations** (fill in as applicable).
- **Models:** one model family and small models (2B main), with a hybrid linear-attention architecture.
- **Lens:** single-token concepts only (J-lens limitation). The lens is a first-order, averaged, observational decoder fitted on WikiText.
- **Items:** curated item banks rather than norms (D-007), with small cells (domestic birds, Asia-landlocked). Exp 1 categories are shared across splits (D-009). Numbers have a tokenization confound (D-008).
- **Geometry estimates:** few samples per manifold (D capped at P − 1).
- **Theory assumptions:** Gaussian-tail error and prototype readout.
- **Inference:** no causal interventions unless T-044 was done.

---

## 5. Code and data availability

`[TBD: T-001 / T-063 — repository link, and whether code submission is required.]`
- **Reproduction:** the full sequence is in `docs/RUNBOOK.md`.
- **Results:** every figure is regenerated from `results/` by script (`scripts/make_figures.py`, T-060).
- **Pins:** model checkpoints, lens files, `jlens` commit 581d398, and `uv.lock`.

---

## References

_From `docs/PLAN.md` §12 and `docs/background/`. `[TBD: citation style per T-001; complete volume/page details from the sources.]`_

- Baars, B. J. (1988). *A Cognitive Theory of Consciousness.*
- Belrose, N. et al. (2023). Eliciting latent predictions from transformers with the tuned lens. arXiv:2303.08112.
- Bernardi, S., Benna, M. K., Rigotti, M., Munuera, J., Fusi, S. & Salzman, C. D. (2020). The geometry of abstraction in the hippocampus and prefrontal cortex. *Cell.*
- Blumensath, T. & Davies, M. E. (2008). Gradient pursuits. (Cited in `lens/jspace.py` for the J-space decomposition algorithm.)
- Chung, S., Lee, D. D. & Sompolinsky, H. (2018). Classification and geometry of general perceptual manifolds. *Physical Review X.*
- Dehaene, S., Kerszberg, M. & Changeux, J.-P. (1998). A neuronal model of a global workspace in effortful cognitive tasks. *PNAS.*
- Flesch, T., Juechems, K., Dumbalska, T., Saxe, A. & Summerfield, C. (2022). Orthogonal representations for robust context-dependent task performance in brains and neural networks. *Neuron.*
- Freedman, D. J., Riesenhuber, M., Poggio, T. & Miller, E. K. (2001). Categorical representation of visual stimuli in the primate prefrontal cortex. *Science.*
- Gurnee, W., Sofroniew, N., Pearce, … Batson, J. & Lindsey, J. (2026). Verbalizable representations form a global workspace in language models. arXiv:2607.15495. Code: github.com/anthropics/jacobian-lens; lenses: huggingface.co/neuronpedia/jacobian-lens.
- Hendel, R., Geva, M. & Globerson, A. (2023). In-context learning creates task vectors.
- Hewitt, J. & Liang, P. (2019). Designing and interpreting probes with control tasks. arXiv:1909.03368.
- Kriegeskorte, N., Mur, M. & Bandettini, P. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience.*
- Mante, V., Sussillo, D., Shenoy, K. V. & Newsome, W. T. (2013). Context-dependent computation by recurrent dynamics in prefrontal cortex. *Nature.*
- Min, S. et al. (2022). Rethinking the role of demonstrations: what makes in-context learning work?
- nostalgebraist (2020). Interpreting GPT: the logit lens.
- Rigotti, M. et al. (2013). The importance of mixed selectivity in complex cognitive tasks. *Nature.*
- Sorscher, B., Ganguli, S. & Sompolinsky, H. (2022). Neural representational geometry underlies few-shot concept learning. *PNAS* 119(43): e2200800119. Code: github.com/bsorsch/geometry-fewshot-learning.
- Todd, E. et al. (2024). Function vectors in large language models. *ICLR.*
- Wei, J. et al. (2023). Larger language models do in-context learning differently. arXiv:2303.03846.
- Zhao, Z. et al. (2021). Calibrate before use: improving few-shot performance of language models. *ICML.*

_Add only if used: Hebart et al. (2019) THINGS; McRae et al. (2005); Devereux et al. (2014) CSLB; Binder et al. (2016); Monsell (2003) task switching; Landau, Smith & Jones (1988) shape bias (Exp 4)._
