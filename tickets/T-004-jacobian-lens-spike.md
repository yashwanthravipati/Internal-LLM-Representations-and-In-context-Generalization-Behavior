---
id: T-004
title: Jacobian lens availability & compatibility spike
status: review
phase: 0
priority: P0
depends_on: [T-003]
owner: claude
estimate: M
tags: [jlens, models, risk]
updated: 2026-10-26
---

# T-004 — Jacobian lens availability & compatibility spike

## Context
The J-lens is central to Exp 3 and decides the primary model, so this is an early risk spike. HF `neuronpedia/jacobian-lens` hosts pre-fitted **J++** lenses (fitted on WikiText-103) for Qwen3.5-0.8B, Qwen3.5-2B, Qwen3-4B and others. We haven't verified (a) how J++ differs from the paper's J-lens, (b) which checkpoint (base vs post-trained) each lens was fitted on, or (c) the hook-point conventions. A lens is only valid for the exact checkpoint it was fitted on.

## Tasks
- [ ] Install `jlens` from github.com/anthropics/jacobian-lens at a **pinned commit** (the repo is unmaintained). Record the commit in `pyproject.toml`/DECISIONS.
- [ ] Load the J++ lens for Qwen3.5-0.8B and Qwen3-4B (at least). Inspect the stored metadata: model ID, `n_prompts`, layers covered, dtype. (HF discussion #3 reports a pre-finalization file for qwen3-32b, so check that the files we use are final.)
- [ ] Find out what "J++" is (Neuronpedia blog, HF card, any write-up by Ayonrinde & Lindsey). Summarize how it differs from the J-lens in `docs/background/jacobian_lens.md`, and resolve the **[verify]** items there.
- [ ] Sanity check: reproduce a paper-style latent readout with `lens.apply`. Example: "Fact: The currency used in the country shaped like a boot is" should show "Italy"-like content in middle layers. Save the slice/top-k output.
- [ ] Check whether `jlens` exposes (a) concept vectors `W_U J_ℓ`, (b) the sparse J-space decomposition (gradient pursuit, k ≤ 25), and (c) steering/patching helpers. Note the API, or what we'll have to implement.
- [ ] Determine the hook point and layer indexing `jlens` uses (residual stream before or after each block; final norm), so our cached activations (T-020) match.
- [ ] Time a 5-prompt `jlens.fit` on Qwen3.5-0.8B and extrapolate to 100 and 1000 prompts (input for T-045).
- [ ] Run a toy few-shot categorization prompt on the lens-matched checkpoint (raw completion). Does it do the task?
- [ ] Recommend the primary model and lens type, and update D-003 with the evidence.

## Acceptance criteria
- [ ] Written recommendation (in the Work log and D-003) backed by evidence: lens loads, readout reproduces, checkpoint match confirmed, few-shot behavior OK.
- [ ] A notebook or script (`notebooks/t004_jlens_spike.ipynb` or `scripts/jlens_sanity.py`) reproduces the readout.
- [ ] `docs/background/jacobian_lens.md` updated with J++ details and the hook conventions.

## Outputs
- DECISIONS D-003 (accepted or revised), updated background doc, sanity script/notebook.

## Notes & pitfalls
- If the pre-fitted lens is for a post-trained checkpoint, few-shot prompting should still work with raw completion. Verify this rather than assume it.
- If J++ differs in ways that matter for our claims (e.g. it isn't an average Jacobian), plan to fit a true J-lens on a small model (T-045).

## References
- `docs/background/jacobian_lens.md`; Gurnee et al. 2026 (arXiv:2607.15495); https://huggingface.co/neuronpedia/jacobian-lens

## Work log
- 2026-10-26 (claude): Installed `jlens` pinned to 581d398613e5602a5af361e1c34d3a92ea82ba8e (its only public commit; `pyproject.toml` `[tool.uv.sources]`). Neuronpedia's 2B fit records commit 4430656…, which isn't public (likely a fork); the lens file format is identical.
  - **Key finding:** `neuronpedia/jacobian-lens` also hosts **original J-lens fits** (not just J++), including `qwen3.5-2b-pt` = **Qwen/Qwen3.5-2B-Base**, `qwen3.5-0.8b` (post-trained), `qwen3.5-2b`, `qwen3.5-4b`, `qwen3.5-9b-pt`. Each `config.yaml` names the exact checkpoint, which settles base vs post-trained. J++ = Ayonrinde & Lindsey (2026), a "drop-in replacement… more coherent on earlier layers"; details in `docs/background/jacobian_lens.md`.
  - Loaded the 0.8B and 2B-Base lenses: `n_prompts` 233 / 335 (= `prompts_fitted`, final early-stopped fits), source layers 0–22, fp16, d = 1024 / 2048.
  - `scripts/jlens_sanity.py` → `results/20261026-0112_t004_qwen3.5-0.8b_jlens-sanity/` and `results/20261026-0112_t004_qwen3.5-2b-base_jlens-sanity/report.json`. J-lens mid-layer readouts are coherent (boot prompt: "currency" from L15–16, "euros" late in 2B-Base; logit lens is noise in the middle layers). Neither model shows "Italy" in the top 6, which isn't surprising at this size. On the toy few-shot prompt the decision position shows the label words from L15 on.
  - Hook convention: block outputs; our `states[l + 1]` == jlens layer l (max |Δ| = 0.0). `jlens` provides no J-space decomposition, concept-vector or steering helpers, so we implement those (T-040/T-042/T-044).
  - Toy few-shot on lens-matched checkpoints works with raw completion: 2B-Base p(correct) = 0.69, label mass 0.88; 0.8B 0.33 / 0.62.
  - Fit timing: a 1-prompt `jlens.fit` on 0.8B (CPU, dim_batch 32) **did not finish in > 45 min** and was stopped. Fitting our own lens (T-045) is not feasible in the agent sandbox; it would need a GPU (Neuronpedia's fits ran on a B200).
  - **Recommendation (D-003 revised):** main = Qwen3.5-2B-Base + plain J-lens; dev = Qwen3.5-0.8B + plain J-lens; J++ only as an optional robustness check.
- 2026-10-26 (claude): Work complete and verified; status held at `review` only because dependency T-003 is held at review (see there).
