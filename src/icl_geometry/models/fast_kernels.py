"""CPU speed patch for Qwen3.5 Gated DeltaNet blocks.

Without the `causal_conv1d` package, transformers falls back to a depthwise `F.conv1d`
(groups = channels), which PyTorch's CPU backend runs as one tiny conv2d per channel and
batch element (~100k calls per forward pass; ~80% of runtime). The depthwise causal conv
with a kernel of 4 is just a weighted sum of 4 shifted copies of the input, so we compute it
that way. `tests/test_fast_kernels.py` checks it matches `F.conv1d`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.activations import ACT2FN


def depthwise_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
    **kwargs,
) -> torch.Tensor:
    """Drop-in for `modeling_qwen3_5.causal_conv1d_fn`.

    hidden_states: [B, C, T]; weight: [C, K]; returns [B, C, T]."""
    _, _, seq_len = hidden_states.shape
    k = weight.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype), (k - 1, 0))
    out = weight[None, :, 0, None] * x[..., :seq_len]
    for j in range(1, k):
        out = out + weight[None, :, j, None] * x[..., j : j + seq_len]
    if bias is not None:
        out = out + bias[None, :, None]
    if activation is not None:
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def patch_qwen3_5_conv() -> bool:
    """Replace the depthwise conv fallback in the Qwen3.5 modeling module. Returns True if
    patched (False if the module is unavailable)."""
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    except ImportError:
        return False
    mod.causal_conv1d_fn = depthwise_causal_conv1d
    return True
