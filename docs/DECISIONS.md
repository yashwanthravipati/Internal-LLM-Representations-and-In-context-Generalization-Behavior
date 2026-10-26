# Decision log

Lightweight decision records. Add an entry whenever a choice affects the science (what's measured, how, on what data) or the architecture. Never delete entries: mark them `superseded by D-NNN` instead.

**Statuses:** `proposed` (needs the user's sign-off or evidence from a ticket) · `accepted` · `superseded`.

Format:

```
## D-NNN — Title
- Date: YYYY-MM-DD · Status: proposed | accepted | superseded · Ticket: T-NNN
- Decision: …
- Why: …
- Alternatives considered: …
```

---

## D-001 — Project choice: Course Project 5 plus two extensions
- Date: 2026-10-25 · Status: **accepted** (user) · Ticket: —
- Decision: Pursue Project 5 ("Can the geometry of internal LLM representations predict in-context generalization behavior?"), extended with (a) context-dependent regrouping of the same items and (b) Anthropic's Jacobian lens.
- Why: The student's choice. Project 5 is pre-approved; the extensions deepen the neuroscience link (Mante/Flesch; global workspace).

## D-002 — File-based ticket system
- Date: 2026-10-25 · Status: **accepted** · Ticket: —
- Decision: Tickets are markdown files in `tickets/` with frontmatter, managed by `scripts/tickets.py` (stdlib only). `tickets/BOARD.md` is generated.
- Why: Works for both humans and agents, lives in the repo, needs no external service, and diffs cleanly in git.

## D-003 — Primary model family: small Qwen3.5 models with pre-fitted lenses
- Date: 2026-10-25, revised 2026-10-26 · Status: **proposed** (evidence from T-003/T-004; accept after the T-016 pilot) · Ticket: T-004
- Decision (revised): **Main model: Qwen/Qwen3.5-2B-Base** with the plain Jacobian lens `qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt` from `neuronpedia/jacobian-lens`. **Dev model: Qwen/Qwen3.5-0.8B** (post-trained; no base-model lens exists at this size) with `qwen3.5-0.8b/jlens/…/Qwen3.5-0.8B_jacobian_lens.pt`. **Scaling candidates (T-051):** Qwen3.5-4B (post-trained; jlens available) and Qwen3.5-9B-Base (`qwen3.5-9b-pt/jlens/…`), outside the sandbox on MPS or a GPU.
- Why:
  - The HF lens repo now also hosts **original J-lens fits** (Anthropic's `jlens` code, WikiText-103, 1000-prompt budget with early stopping), not only J++. Using them removes the "J++ ≠ J-lens" risk for our claims; J++ becomes an optional robustness check.
  - A **base** checkpoint suits raw few-shot completion, and its lens was fitted on exactly that checkpoint (config `hf_model_name: Qwen/Qwen3.5-2B-Base`).
  - Evidence: on the 4-shot toy prompt, 2B-Base puts 0.88 of its mass on the two labels (0.69 correct) vs 0.62 (0.33 correct) for 0.8B. Both lenses load, our residual convention matches `jlens` exactly, and the J-lens gives coherent mid-layer readouts where the logit lens is noise (`results/20261026-0112_t004_*`).
  - Fits the hardware: 2B-Base needs ~13 GB RAM in fp32 on CPU and runs ~2 prompts/s (`docs/COMPUTE.md`).
- Open: whether 2B-Base has enough dynamic range on Exp 2 (T-016). Qwen3.5 is a hybrid linear-attention architecture; note it in the write-up, since the J-lens paper's open-model demos were on standard transformers.

## D-004 — Three-experiment structure with Exp 1 as the MVP
- Date: 2026-10-25 · Status: **proposed** (confirm with user) · Ticket: —
- Decision: Exp 1 (neutral geometry → ICL, the core Project 5) must land before Exp 2 (regrouping) and Exp 3 (J-lens) get significant effort. P2 tickets wait until milestone M2.
- Why: It guarantees a complete, on-brief result even if the extensions fail. The extensions roughly raise the difficulty from 5 to 7.

## D-005 — Primary metrics locked before analysis (lightweight pre-registration)
- Date: 2026-10-25 · Status: **proposed**; lock before T-024 runs · Ticket: T-024
- Decision: The primary metrics are those in PLAN §6.9. Anything else is exploratory.
- Why: Many layers × metrics × models create a big garden of forking paths.

## D-006 — Positions and stream
- Date: 2026-10-25 · Status: **proposed** · Ticket: T-020
- Decision: Read the residual stream after each block (plus embeddings) at two positions: `query_item` (the last subtoken of the query item) and `decision` (the final prompt token).
- Why: `query_item` tests whether the item's representation shifts (H2a); `decision` tests readout selection (H2b). Hook conventions must match `jlens` (T-004).

## D-009 — Held-out splits fixed before analysis
- Date: 2026-10-25, procedure specified 2026-10-26 · Status: **proposed** (confirm) · Ticket: T-014
- Decision: Items are split into **val (40%) / test (60%)** once, stratified by category (Exp 1) or crossed cell (Exp 2), by sorting each stratum's items on a salted SHA-256 hash of the item string (`icl_geometry.tasks.prompts.assign_splits`, salt `icl-geometry-split-v1`). The split is stable across machines and runs, and doesn't depend on row order. Every prompt set draws demos and queries from one split only, so val and test prompts share no items. Layer and hyperparameter selection use val; headline numbers use test. Pilot prompt sets use val only.
- Why: Avoids double dipping when choosing the "best layer", and keeps item identity from leaking across the split.
- Caveat: Exp 1 shares *categories* across splits (only items differ). If a category-level split is needed for task-level claims, it can be layered on top (pairs whose categories are both in a held-out category set).

## D-013 — Primary theory variant: full SNR expression
- Date: 2026-10-26 · Status: **proposed** (lock with D-005 before T-024) · Ticket: T-022
- Decision: The primary theory prediction uses Eq. 1 with the official code's normalization (signal / tr(C_a)), the subspace–subspace overlap term, and the SI 1/m² terms (`snr(..., full=True)`). The dominant-terms version is reported as a sensitivity check.
- Why: On synthetic manifolds the full version is closer to Monte Carlo at small m (high-D MAE 0.005 vs 0.016 at m = 1); the two converge for m ≥ 5. Our m grid starts at 1.
- Alternatives: dominant terms only (as in the paper's figures).

## D-007 — Task domains
- Date: 2026-10-26 · Status: **proposed** (needs the student's sign-off: T-010 is in `review`) · Ticket: T-010
- Decision:
  - **Exp 1:** 31 curated categories under 7 superordinates (animal, plant, food, artifact, body, person, nature); 970 items; 465 category pairs (95 within-superordinate = "hard" pairs, 370 across). Relatedness = shared superordinate. Items are lowercase singular common nouns, each in one category; polysemous or debatable items are marked `medium`/`low` and left out of main analyses.
  - **Exp 2 (4 domains):** **animals** (mammal/bird × domestic/wild), **countries** (Europe/Asia × coastal/landlocked), **numbers** (even/odd × < 50 / > 50, two-digit, knowledge-free control), **words** (animal/food × first letter b/p; surface vs semantic).
  - **Dropped: foods (fruit/vegetable × red/green).** Colour isn't objective for most produce (apples, grapes and peppers come in several colours), and red vegetables are scarce once tomato is excluded. The words domain covers the "food" semantics instead.
  - Label sets: foo/bar (default), dax/wug, zip/zap; plus a meaningful-label control (the attribute value names). Attribute words for the J-lens readout are in `configs/domains.yaml`.
- Why objective and independent: continent/coast, parity/magnitude and category/first-letter are factual and statistically independent by construction (cells filled from both values of each attribute). Class and domestication are independent (domestic and wild members in both classes), but domestication is the least objective attribute: only 6 domestic birds are unambiguous.
- Known weaknesses: domestic-bird cell (6 high-confidence items), Asia-landlocked (8). Countries are proper nouns, and some are multi-token. Source is project curation, not THINGS/McRae norms; a cross-check against THINGS or a second rater is still open (T-011).
- Alternatives considered: THINGS categories (licence/download from OSF not attempted from the sandbox); foods by colour (rejected above); animals by habitat (water/land) or diet (both fuzzier than domestication).

## D-008 — Tokenization policy
- Date: 2026-10-26 · Status: **proposed** · Ticket: T-012
- Decision:
  - Multi-token items are allowed and represented by their **last subtoken** at the `query_item` position; `n_tokens` is stored per item and template and used as a covariate. A sensitivity analysis is rerun on the single-token subset.
  - Labels and J-lens attribute words must be single tokens with a leading space. Replacements: `dax/wug` → `dax/mip` (" wug" is 2 tokens); attribute words " mammal" and " landlocked" (2 tokens each) dropped in favour of " mammals"/" fur" and " inland"/" land"; meaningful labels per attribute are in `configs/domains.yaml`.
  - **Template matters:** items starting a line (`arrow` template) are single tokens only ~22% of the time; items after a space (`io` template: `Input: dog\nOutput: foo`) ~53% (countries 85%). Template choice is left to the pilot (D-010), with tokenization as one criterion.
  - Numbers are always 2 tokens (one per digit). The last subtoken is the units digit, which carries parity directly but not magnitude, a confound for `query_item` geometry in the numbers domain; noted for T-032.
- Why: Dropping multi-token items would remove most of the bank; last-subtoken representations are standard.
- Facts: Qwen3.5-0.8B and Qwen3.5-2B-Base share an identical vocabulary, so one token table serves both (tables are still written per model). No item changes the separator's tokenization.

## D-011 — Neutral-representation protocol
- Date: 2026-10-26 · Status: **proposed** (student sign-off; T-021 in review) · Ticket: T-021
- Decision: Neutral representation = state at the item's last subtoken, averaged over 4 carriers (`icl_geometry.models.neutral.CARRIERS`): "The word is {item}", "Here is one example: {item}", "Item: {item}", "Next line:\n{item} ->". The item is never at position 0 (Qwen has no BOS; position 0 is an attention sink). The carriers mix the leading-space and line-initial tokenizations. Per-carrier states are cached for the sensitivity analysis (carriers as extra manifold samples).
- Evidence (0.8B): carriers agree well (mean pairwise cosine 0.81 at the embeddings, 0.87–0.93 at layers 3–24).

## D-012 — Outlier-dimension handling
- Date: 2026-10-26 · Status: **proposed** · Ticket: T-021
- Decision: **Primary: raw residual states (no standardization or whitening)**, because the theory's quantities are defined in the raw Euclidean space. Sensitivity analysis: drop the top-5 dimensions by mean |activation|, and per-dimension standardization.
- Why: At the item positions used here, no dimension dominates: the top-1 dimension holds ≤ 4% of the variance at every layer on 0.8B (`results/neutral_reps/qwen3.5-0.8b/`). Massive activations are a first-token / delimiter phenomenon and don't show up at item tokens. Re-check on 2B-Base before locking.

<!-- Reserved numbers for decisions the tickets will make:
D-007 domains (T-010) · D-008 tokenization policy (T-012) · D-010 final template/labels/m grid (T-016)
D-011 neutral-template aggregation (T-021) · D-012 outlier-dimension handling (T-021) -->
