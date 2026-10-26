"""Model loading and residual-stream access.

Residual convention (matches `jlens`, see docs/background/jacobian_lens.md): the state at
block index l is the *output of block l* (a forward hook on `model.model.layers[l]`), so the
final entry is the last block's output *before* the final norm. We also keep the token
embeddings as state 0, giving `n_layers + 1` states per position:

    states[0]     = embeddings           (HF hidden_states[0])
    states[l + 1] = output of block l    (HF hidden_states[l + 1], except the last one,
                                          which HF returns after the final norm)

So `jlens` layer l  <->  our state index l + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import transformers

from icl_geometry.models.fast_kernels import patch_qwen3_5_conv


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    return getattr(torch, dtype)


@dataclass
class LoadedModel:
    model_id: str
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype

    @property
    def text_config(self):
        return self.model.config.get_text_config()

    @property
    def n_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def d_model(self) -> int:
        return self.text_config.hidden_size

    @property
    def blocks(self) -> torch.nn.ModuleList:
        return self.model.model.layers

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Final norm + LM head (same as `jlens.HFLensModel.unembed`)."""
        w = self.model.lm_head.weight
        return self.model.lm_head(self.model.model.norm(residual.to(w.device, w.dtype)))


def load_model(model_id: str, dtype: str = "auto", device: str = "auto") -> LoadedModel:
    dev = resolve_device(device)
    dt = resolve_dtype(dtype, dev)
    patch_qwen3_5_conv()  # exact; ~15x faster forward passes on CPU (see fast_kernels.py)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dt)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    # Right padding: with a causal model, pad tokens after a sequence cannot affect it, which
    # is also safe for the linear-attention (Gated DeltaNet) blocks in Qwen3.5.
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return LoadedModel(model_id, model, tok, dev, dt)


class ResidualRecorder:
    """Forward hooks capturing block outputs (the jlens convention) for all blocks."""

    def __init__(self, blocks: torch.nn.ModuleList):
        self.blocks = blocks
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> ResidualRecorder:
        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_hook(self._hook(i)))
        return self

    def _hook(self, i: int):
        def fn(module, inputs, output):
            self.outputs[i] = (output if torch.is_tensor(output) else output[0]).detach()

        return fn

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def forward_with_states(
    lm: LoadedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (logits [B, T, V], states [B, T, n_layers + 1, d_model]) in the convention above."""
    with ResidualRecorder(lm.blocks) as rec:
        out = lm.model(
            input_ids=input_ids.to(lm.device),
            attention_mask=None if attention_mask is None else attention_mask.to(lm.device),
            output_hidden_states=True,
            use_cache=False,
        )
    emb = out.hidden_states[0]
    states = torch.stack([emb] + [rec.outputs[i] for i in range(lm.n_layers)], dim=2)
    return out.logits, states
