# The Jacobian lens (J-lens)

_Notes compiled 2026-10-25 from the paper, the official repository and the Hugging Face lens repository. Items marked **[verify]** were not confirmed and are assigned to T-004._

## Source

- **Paper:** Gurnee, Sofroniew, Pearce, Piotrowski, Kauvar, Chen, Soligo, Bogdan, Ong, Wang, Thompson, Abrahams, Kantamneni, Ameisen, Batson & Lindsey (Anthropic, 2026). *Verbalizable Representations Form a Global Workspace in Language Models.* arXiv:2607.15495.
- **Code:** https://github.com/anthropics/jacobian-lens (the `jlens` package; Apache-2.0; **marked not maintained, not accepting contributions**).
- **Pre-fitted lenses:** https://huggingface.co/neuronpedia/jacobian-lens · demo: https://www.neuronpedia.org/jlens

## What it is

The J-lens reads out what an internal activation is **disposed to make the model say**, at any layer and position. It linearly transports a residual-stream vector into the final-layer basis using the **average input–output Jacobian**, then decodes it with the model's own unembedding:

```
J_ℓ = E[ ∂h_final,t′ / ∂h_ℓ,t ]        expectation over prompts, source positions t, and all t′ ≥ t
lens_ℓ(h) = softmax( W_U · norm( J_ℓ h ) )
```

- J_ℓ is a d_model × d_model matrix per layer, fit on a pretraining-like corpus. The paper uses 1,000 sequences of 128 tokens; quality saturates quickly, and **~100 prompts is usable**.
- Compared with the **logit lens** (J = I), the J-lens accounts for how later layers transform a direction, so it reads mid-layer content more reliably. The paper notes the logit lens often captures similar structure but is less reliable in early layers.

## Concept vectors, J-space, interventions

- **Concept (J-lens) vector** for vocabulary token t at layer ℓ: v_t = J_ℓᵀ W_U[t], one row of W_U J_ℓ. The probe score is ⟨v_t, h⟩, or the cosine similarity.
  - Because of the norm(·) in the readout, ⟨v_t, h⟩ is a linear approximation. That's fine for ranking concepts at one position; be careful when comparing raw scores across positions or layers.
- **J-space:** the set of points expressible as a **sparse non-negative combination** of J-lens vectors, with sparsity k ≤ 25 (about the number of vectors meaningfully active at once). Geometrically it is a union of k-dimensional cones. It is decomposed by **gradient pursuit**. The J-space component explains a small share of activation variance (≤ ~10%), yet it carries the "workspace" content.
- **Interventions** described in the paper:
  - steering: h ← h + α·v_t;
  - ablation: project out v_t;
  - patching in lens coordinates: with V = [v_s, v_t] and c = V⁺h, set h' = h + V(σ(c) − c), where σ swaps the two coordinates. This leaves everything orthogonal to span{v_s, v_t} untouched.

## Findings that matter for us

- **Coherent content only in an intermediate band of layers.** Readouts are noisy in roughly the first third of the network, and the last few layers switch to representing the next output token. Expect our rule-readout effects (H4) in the middle band.
- **Implicit properties appear only when the task asks for them.** Part of speech or verb tense shows up in the J-space only when the question asks for it, even though next-word predictions respect the property either way. **This is the closest precedent for our Exp 2/3.** We test whether *demonstrations alone* (with no explicit question) recruit the cued attribute ("mammal" vs "wild") into the workspace.
- **Unspoken intermediates:** two-hop facts, planned rhymes and arithmetic intermediates are readable before they're said.
- **Flexible generalization through swaps:** swapping one country's lens vector for another's shifts answers to capital, language and continent questions. Success varies by category, and number words are weakest. This supports the causal stretch goal (T-044).
- **Layer indexing:** the paper reports 25 evenly spaced layers re-indexed to 0–100. We adopt normalized depth for cross-model plots.
- The main models are Claude Sonnet 4.5 (with Haiku 4.5, Opus 4.5/4.6). Open-weight readouts are on Neuronpedia.

## Limitations the authors note

- Only concepts that map to **single vocabulary tokens** are captured (multi-token extensions are in the paper's §A.9). Our attribute words and labels must be single tokens (T-012).
- It's a **first-order, averaged** linear approximation ("imperfect", "approximate").
- The decomposition is non-unique (the vectors are overcomplete).
- Intervention effects are imperfect and sensitive to prompt phrasing.
- Lenses are **observational decoders**. Causal claims need interventions (T-044).

## Using the code

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

# pre-fitted
lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(model, "Fact: The currency used in the country shaped like a boot is", positions=[-2])

# fit our own (slow; dominated by backward passes; parallelize with JacobianLens.merge)
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

The repo also ships `walkthrough.ipynb`, an interactive layer × position "slice" visualization, and Anthropic-authored replication/evaluation prompt sets under `data/`.

## Pre-fitted lenses for open models (as of 2026-10-25)

`neuronpedia/jacobian-lens` hosts **"J++" lenses** in `<model>/jpp/Salesforce-wikitext/`, fitted on WikiText-103. The card also mentions a `<model>/jlens/` folder but lists no models in it.

| Model | File | Fitted by | Recall@10 |
|---|---|---|---|
| Qwen/Qwen3.5-0.8B | `qwen3.5-0.8b/jpp/Salesforce-wikitext/Qwen3.5-0.8B_jpp_lens.pt` | Neuronpedia | 44.9% |
| Qwen/Qwen3.5-2B | `qwen3.5-2b/jpp/Salesforce-wikitext/Qwen3.5-2B_jpp_lens.pt` | Neuronpedia | 51.5% |
| Qwen/Qwen3-4B | `qwen3-4b/jpp/Salesforce-wikitext/Qwen3-4B_jpp_lens.pt` | Neuronpedia | 53.8% |
| Qwen/Qwen3.5-9B | `qwen3.5-9b/jpp/Salesforce-wikitext/Qwen3.5-9B_jpp_lens.pt` | Ayonrinde & Lindsey | 55.0% |
| Larger (Qwen3.6-27B, gemma-4-31B, Olmo-3-32B, Qwen3.5-122B-A10B, DeepSeek-V4-Flash) | see card | Ayonrinde & Lindsey | 48–61% |

Notes:

- The card says the Neuronpedia and Ayonrinde & Lindsey scores were computed differently and may not be directly comparable.
- Known issue (HF discussion #3): the `qwen3-32b` file appears to be a pre-finalization checkpoint. **Check the stored `n_prompts` in any file we use.**
- A community repo, `solarkyle/jspace-lenses`, has lenses for Gemma 4 variants and Qwen3.6-27B (Gemma license applies).
- A lens is valid **only for the exact checkpoint it was fitted on**. A mismatch gives shape errors or plausible-looking nonsense.

### Resolved in T-004 (2026-10-26)

- **Plain J-lens fits exist too.** Besides the J++ files, the repo has original J-lens fits in `<model>/jlens/Salesforce-wikitext/` (fitted by Neuronpedia with Anthropic's `jlens` code, WikiText-103, `n_prompts` budget 1000 with early stopping at mean relative change < 0.002). Relevant to us: `qwen3.5-0.8b` (233 prompts fitted), `qwen3.5-2b` (283), **`qwen3.5-2b-pt` = Qwen/Qwen3.5-2B-Base (335)**, `qwen3.5-4b` (417; also an `_n1000` file by Mateusz Piotrowski), `qwen3.5-9b-pt` = 9B-Base, `qwen3-1.7b`, `qwen3-4b`, `qwen3-8b`, and others. Each folder has a `config.yaml` with the exact command and `hf_model_name`, which settles base vs post-trained: `qwen3.5-0.8b` → post-trained `Qwen/Qwen3.5-0.8B`; `qwen3.5-2b-pt` → `Qwen/Qwen3.5-2B-Base`.
- **What J++ is.** *J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses* (Ayonrinde & Lindsey, 2026), described on the card as "an improved, drop-in replacement for the Jacobian Lens… more coherent on earlier layers." It builds on the J-lens and on the R-Lens (Blank & Bhatia). The Qwen3.5-0.8B J++ config shows 96 WikiText prompts, `lrp_mode: rlens`, target layer 23, `skip_first: 16`, `capture_point: block_output`. The file format is the same `{J, source_layers, d_model, n_prompts, provenance}` dict, applied as `unembed(h @ J[l].T)`. We use the **original J-lens** for primary claims (D-003); J++ is optional for robustness (available for post-trained Qwen3.5-0.8B/2B, not for 2B-Base).
- **Hook convention.** `jlens` records the **output of each decoder block** (forward hook on `layers[l]`), source layers 0…n_layers−2, target = the last block's output; readout = `lm_head(final_norm(J_l h))`. HF's last `hidden_states` entry has the final norm already applied, so we extract with hooks (`icl_geometry.models.loading`), giving states[0] = embeddings and states[l+1] = block l output. Verified: max |difference| from `jlens.ActivationRecorder` = 0.0 on both models.
- **The pinned `jlens` (commit 581d398, its only public commit) does not include** J-space sparse decomposition (gradient pursuit), concept-vector helpers, or steering/patching. We implement these ourselves (T-040, T-042, T-044). It does provide `JacobianLens.apply/transport/merge`, `fit`, `ActivationRecorder` and a slice visualiser.
- **Pre-finalization files.** Lens files from the Neuronpedia fit store no `provenance` key, only `n_prompts`, which matches `results.prompts_fitted` in each `config.yaml` (e.g. 233 for 0.8B), so they are the final early-stopped fits.
- **Sanity readouts** (`scripts/jlens_sanity.py`; `results/20261026-0112_t004_*`): on "Fact: The currency used in the country shaped like a boot is", both models' J-lens shows "currency/currencies" from layer 15–16 on and "euros" late (2B-Base, layers 20–22); the logit lens is mostly noise in middle layers. Neither small model shows "Italy" in the top 6. On the toy few-shot prompt, the J-lens decision position reads " ->" until ~layer 14 and the label words (" foo", " bar") from layer 15 on; the query-item position reads the item itself ("wolf") through middle layers. Category words are not in the top 6 there, so H4 must be tested with lens **scores/ranks** of attribute words, not top-k membership.
- **Tokenization note.** Items that start a line tokenize without a leading space (`\nwolf` → `wolf`); labels after `->` keep one (` foo`).

## How we use it (Exp 3)

1. **Rule readout (H4, T-041):** score the cued vs uncued attribute words at the query-item and decision positions across layers. Positive control: explicit-question prompts. Negative control: random-label prompts.
2. **J-space geometry (H5, T-042/T-043):** recompute the Sorscher SNR and shift metrics on J-space components, sparse codes, and attribute-word spans. Compare with full hidden states **at matched dimensionality** (random subspaces with the Eq. 2 null, PCA, probe directions, logit lens).
3. **Causal test (stretch, T-044):** steering, ablation and lens-coordinate patching to flip the grouping.
