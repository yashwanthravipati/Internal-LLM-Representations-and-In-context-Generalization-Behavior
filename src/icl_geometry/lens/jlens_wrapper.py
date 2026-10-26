"""Minimal Jacobian-lens interface over cached residual states (T-040).

Depends on `jlens` only for the lens file format (a dict with `J`); everything else is
reimplemented here so it works on cached activations.

Layer indexing: lens layer l transports the output of decoder block l, which is our cached
state index l + 1 (states[0] = embeddings). Use `state_index(l)` when slicing caches.

Readout (identical to `jlens.JacobianLens.apply`):
    logits = lm_head(final_norm(J_l @ h))

Concept vectors. The final norm is RMSNorm with an effective per-dimension gain g (for
Qwen3.5 the stored weight is zero-centred and the gain is 1 + w; we measure g numerically as
norm(1)). Then lm_head(norm(x)) = W_U (g * x) / rms(x), so
    v_t = J_l^T (g * W_U[t])   and   <v_t, h> = rms(J_l h) * lens_logit_t(h).
Within one position the ranking of <v_t, h> over tokens is exactly the lens ranking. Across
positions or layers, compare z-scores or ranks, not raw scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from huggingface_hub import hf_hub_download

from icl_geometry.models.loading import LoadedModel

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILES = {
    "Qwen/Qwen3.5-0.8B": "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
    "Qwen/Qwen3.5-2B-Base": (
        "qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"
    ),
    "Qwen/Qwen3.5-2B": "qwen3.5-2b/jlens/Salesforce-wikitext/Qwen3.5-2B_jacobian_lens.pt",
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
}


def state_index(layer: int) -> int:
    """Lens layer l -> cached state index (block l output)."""
    return layer + 1


@dataclass
class Lens:
    model_id: str
    J: dict[int, torch.Tensor]  # layer -> [d, d] float32
    n_prompts: int
    identity: bool = False  # logit lens (J = I)

    @property
    def layers(self) -> list[int]:
        return sorted(self.J)

    @classmethod
    def load(cls, model_id: str, path: str | None = None) -> Lens:
        path = path or hf_hub_download(LENS_REPO, LENS_FILES[model_id])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        return cls(model_id, {int(k): v.float() for k, v in ckpt["J"].items()},
                   int(ckpt["n_prompts"]))

    @classmethod
    def logit_lens(cls, model_id: str, n_layers: int, d_model: int) -> Lens:
        eye = torch.eye(d_model)
        return cls(model_id, {layer: eye for layer in range(n_layers - 1)}, 0, identity=True)

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        """J_l h for h [..., d]."""
        if self.identity:
            return h.float()
        return h.float() @ self.J[layer].T.to(h.device)


def norm_gain(lm: LoadedModel) -> torch.Tensor:
    """Effective per-dimension gain of the final RMSNorm: norm(ones) (rms(ones) = 1)."""
    w = lm.model.lm_head.weight
    with torch.no_grad():
        return lm.model.model.norm(torch.ones(1, lm.d_model, dtype=w.dtype, device=w.device))[
            0].float().cpu()


@torch.no_grad()
def readout(lm: LoadedModel, lens: Lens, h: torch.Tensor, layer: int) -> torch.Tensor:
    """Lens logits [..., vocab] for residuals h [..., d] at lens layer `layer`."""
    return lm.unembed(lens.transport(h, layer)).float().cpu()


def top_tokens(lm: LoadedModel, logits: torch.Tensor, k: int = 10) -> list[str]:
    return [lm.tokenizer.decode([int(i)]) for i in logits.topk(k).indices]


@torch.no_grad()
def concept_vectors(lm: LoadedModel, lens: Lens, token_ids: list[int], layer: int) -> torch.Tensor:
    """[n_tokens, d] with v_t = J_l^T (g * W_U[t])."""
    W = lm.model.lm_head.weight[token_ids].float().cpu() * norm_gain(lm)
    return W if lens.identity else W @ lens.J[layer]


def concept_scores(h: torch.Tensor, vectors: torch.Tensor) -> dict[str, torch.Tensor]:
    """Dot products and cosines between residuals h [N, d] and concept vectors [T, d]."""
    h = h.float()
    dot = h @ vectors.T
    cos = dot / (h.norm(dim=-1, keepdim=True) * vectors.norm(dim=-1)[None] + 1e-12)
    return {"dot": dot, "cos": cos}
