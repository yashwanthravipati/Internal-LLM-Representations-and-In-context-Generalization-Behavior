---
id: T-003
title: Compute environment & model access check
status: done
phase: 0
priority: P0
depends_on: [T-002]
owner: claude
estimate: S
tags: [infra, models]
updated: 2026-10-26
---

# T-003 — Compute environment & model access check

## Context
We need to know what hardware we have and that the candidate models load, run few-shot prompts, and expose residual-stream activations before building the pipelines. Candidates (PLAN §4.1): Qwen3.5-0.8B (dev), Qwen3.5-2B and Qwen3-4B (main), Qwen3.5-9B (optional, for scaling). These were chosen because pre-fitted Jacobian lenses exist for them.

## Tasks
- [ ] Record the hardware: GPU model, VRAM, CUDA/driver version, disk/scratch quota.
- [ ] HF authentication; download the candidate models. Use the exact repo IDs and variants (base vs post-trained) that T-004 says the lenses were fitted on. If T-004 hasn't finished, download the names as listed on `neuronpedia/jacobian-lens`.
- [ ] Write `scripts/smoke_test.py`: load in bf16, run a 4-shot categorization prompt (PLAN §4.6 template), print the top-5 next tokens and the probabilities of the two labels.
- [ ] Confirm `output_hidden_states=True` returns `n_layers + 1` states of shape `[batch, seq, d_model]`. Record n_layers and d_model **from each model's config**; don't assume them.
- [ ] Check the architecture: some newer Qwen releases may use non-standard (e.g. hybrid-attention) blocks. Confirm residual-stream access works the same way at every layer, and note any exceptions.
- [ ] Measure throughput (prompts/sec at batch sizes 16–64 with ~100-token prompts) and peak memory. Extrapolate the wall-clock time for a 15k-prompt sweep with activation caching.
- [ ] Propose a dev model and main model(s); add the evidence to D-003 in `docs/DECISIONS.md`.

## Acceptance criteria
- [ ] `docs/COMPUTE.md` exists, with a table: model · params · n_layers · d_model · dtype · peak VRAM · throughput · notes.
- [ ] `uv run python scripts/smoke_test.py --model <id>` runs end to end on at least the dev and main candidates.
- [ ] Storage estimate for activations (formula in PLAN §7.4) filled in with the real dimensions.

## Outputs
- `docs/COMPUTE.md`, `scripts/smoke_test.py`, a DECISIONS D-003 update.

## Notes & pitfalls
- Post-trained ("instruct"/"thinking") checkpoints may emit chat-style tokens. Use **raw completion without a chat template** for few-shot prompts and check that the label mass is high.
- HF's last `hidden_states` entry may already have the final norm applied. Note this; T-020 and T-040 must handle it consistently with `jlens`.

## References
- PLAN §4.1, §7.4; `docs/background/jacobian_lens.md` (pre-fitted lens table)

## Work log
- 2026-10-26 (claude): Hardware is an M5 Pro MacBook (24 GB); MPS unavailable inside the agent sandbox, so agent runs are CPU-only. `scripts/smoke_test.py` + `src/icl_geometry/models/loading.py` (hook-based residual extraction, right padding, unembed). Ran on Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B-Base: both load as `Qwen3_5ForCausalLM`, 24 layers, d = 1024 / 2048; hybrid Gated-DeltaNet/full-attention blocks, hooks work identically on both block types. HF's last hidden state is post-norm (documented). Right-padding invariance holds (≤ 2e-5). Found and fixed a CPU bottleneck (depthwise conv fallback; `models/fast_kernels.py`, tested exact, ~15× speedup). `docs/COMPUTE.md` written with the model table, storage and wall-clock estimates. D-003 updated with evidence (main: 2B-Base; dev: 0.8B).
  - HF auth was not needed (public models). Weights cached under `data/hf_home/` (gitignored) because the default HF cache isn't writable from the sandbox.
- 2026-10-26 (claude): Work complete and verified; status held at `review` only because a dependency (T-002 git init, or T-011 item spot-check) is still open. Flip to `done` once that closes.
- 2026-10-26 (claude): Dependency closed; status → done.
