# Compute environment

_Measured 2026-10-26 (T-003) with `scripts/smoke_test.py`._

## Hardware

| | |
|---|---|
| Machine | MacBook Pro, Apple M5 Pro, 24 GB unified memory, arm64 |
| OS | macOS 26 (Darwin 25.6) |
| GPU | Apple GPU via PyTorch MPS. **Not available inside the Claude Code sandbox** (torch reports "MPS backend is supported on macOS 14.0+", a sandbox artefact). Agent runs therefore use the CPU; runs launched by the user outside the sandbox can use `device: mps`. No CUDA. |
| Disk | ~340 GB free |
| Software | Python 3.12.15 (uv venv), torch 2.14.1, transformers 5.20.2, jlens @ 581d398 |
| Course compute | Unknown until T-001 (the plan assumes one 16 GB GPU) |

## Models (CPU, float32, 5 threads)

| Model | Params | n_layers | d_model | Arch | Peak RSS | Throughput (~100-token prompts) | 4-shot toy prompt: p(correct), label mass |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B (post-trained) | 752 M | 24 | 1024 | hybrid: 3 Gated-DeltaNet (linear attention) : 1 full attention | 7.9 GB | ~4 prompts/s (bs 16) | 0.33 (" foo"), 0.62 |
| Qwen/Qwen3.5-2B-Base | 1.88 B | 24 | 2048 | same hybrid | 13.1 GB | ~2.3 prompts/s (bs 16–32) | 0.69 (" foo"), 0.88 |

The Qwen3.5 checkpoints load through `AutoModelForCausalLM` as the text-only `Qwen3_5ForCausalLM` (`model.model.layers`, `model.model.norm`, `model.lm_head`), so `jlens.from_hf` finds them with the default layout.

**Important speed fix.** Without the `causal_conv1d` package, transformers' Qwen3.5 Gated DeltaNet uses a depthwise `F.conv1d` that PyTorch runs on the CPU as ~100k tiny conv2d calls per forward pass (~80% of runtime). `icl_geometry.models.fast_kernels` replaces it with an exact shifted-sum implementation (max state difference 1.6e-5 in fp32), which made forward passes **~15× faster** (32 prompts × 48 tokens: 40 s → 2.6 s). `load_model` applies the patch automatically. The `chunk_gated_delta_rule` reference implementation is still used (no flash-linear-attention on CPU); it is not the bottleneck.

### Checks (both models)

- `hidden_states` has n_layers + 1 = 25 entries. Entries 1..23 equal the block outputs; **the last entry has the final norm applied** (it equals `norm(block 23 output)`, not the block output). Our extraction therefore uses forward hooks on the blocks, matching `jlens` exactly (max difference 0.0).
- `unembed(final block output)` reproduces the model's logits.
- Right padding leaves real-token states unchanged (max |Δ| ≈ 1e-5), including through the linear-attention blocks. We pad on the right and index positions per row.

## Storage estimate (PLAN §7.4)

bytes ≈ n_prompts × n_positions × (n_layers + 1) × d_model × 2 (fp16)

| Model | per prompt (2 positions, 25 states) | 10k prompts | 15k prompts |
|---|---|---|---|
| Qwen3.5-0.8B (d = 1024) | 102 KB | 1.0 GB | 1.5 GB |
| Qwen3.5-2B-Base (d = 2048) | 205 KB | 2.0 GB | 3.1 GB |

## Wall-clock estimates (CPU, agent sandbox)

Prompts in this project are short (roughly 20–80 tokens), so these are conservative.

| Sweep | 0.8B | 2B-Base |
|---|---|---|
| 15k prompts, forward pass + activation caching | ~1 h | ~2 h |

Using MPS outside the sandbox should be several times faster; measure with `uv run python scripts/smoke_test.py --model <id> --device mps --dtype bfloat16`.

## Jacobian-lens fitting

Not needed for the main plan, because pre-fitted lenses exist (see `docs/background/jacobian_lens.md`). Fit timing on CPU is recorded in the T-004 work log.
