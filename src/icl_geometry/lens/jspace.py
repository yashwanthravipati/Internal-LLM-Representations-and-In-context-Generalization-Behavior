"""J-space: sparse non-negative decomposition of residuals onto J-lens concept vectors (T-042).

Following the J-lens paper, a residual h at lens layer l is approximated by a non-negative
combination of at most k (<= 25) concept vectors v_t = J_l^T (g * W_U[t]) (see
`jlens_wrapper.concept_vectors`), found by greedy gradient pursuit (Blumensath & Davies 2008)
with non-negativity:

    repeat k times:
        pick the atom with the largest positive correlation with the current residual
        take an exact line-search gradient step on the active coefficients, clip at 0

The dictionary is restricted to a *concept vocabulary* (default: tokens that are a leading
space plus a lowercase alphabetic word of length >= 3, plus any required tokens such as the
attribute words), because the full vocabulary (~248k tokens x d) is too large per layer.
Atoms are unit-normalized; codes are on the unit-norm scale.

Outputs per residual: active token ids, coefficients, and the reconstruction (the "J-space
component" of h). `variance_explained` reports the fraction of (centered) variance captured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from icl_geometry.lens.jlens_wrapper import Lens, concept_vectors
from icl_geometry.models.loading import LoadedModel

WORD_RE = re.compile(r"^ [a-z]{3,}$")


def concept_vocabulary(tok, required: list[int] | None = None, max_size: int | None = None
                       ) -> list[int]:
    """Token ids of space-prefixed lowercase words (>= 3 letters), plus `required` ids."""
    ids = [i for t, i in tok.get_vocab().items()
           if WORD_RE.match(tok.convert_tokens_to_string([t]))]
    ids = sorted(set(ids))
    if max_size is not None:
        ids = ids[:max_size]
    if required:
        ids = sorted(set(ids) | set(required))
    return ids


@dataclass
class Dictionary:
    token_ids: list[int]
    atoms: torch.Tensor  # [T, d] unit-norm
    norms: torch.Tensor  # [T] original norms of v_t
    layer: int


def build_dictionary(lm: LoadedModel, lens: Lens, layer: int, token_ids: list[int],
                     chunk: int = 8192) -> Dictionary:
    parts = [concept_vectors(lm, lens, token_ids[i : i + chunk], layer)
             for i in range(0, len(token_ids), chunk)]
    V = torch.cat(parts)
    norms = V.norm(dim=-1)
    return Dictionary(list(token_ids), V / norms[:, None], norms, layer)


@dataclass
class SparseCode:
    indices: torch.Tensor  # [N, k] dictionary row indices (-1 = unused)
    coefs: torch.Tensor  # [N, k] non-negative
    recon: torch.Tensor  # [N, d]

    def token_ids(self, dictionary: Dictionary) -> np.ndarray:
        ids = np.array(dictionary.token_ids)
        idx = self.indices.numpy()
        return np.where(idx >= 0, ids[np.clip(idx, 0, None)], -1)

    def dense(self, n_atoms: int) -> torch.Tensor:
        """[N, T] dense code matrix (vocabulary-indexed features)."""
        out = torch.zeros(self.indices.shape[0], n_atoms)
        valid = self.indices >= 0
        rows = torch.arange(self.indices.shape[0])[:, None].expand_as(self.indices)
        out[rows[valid], self.indices[valid]] = self.coefs[valid]
        return out


@torch.no_grad()
def gradient_pursuit(h: torch.Tensor, atoms: torch.Tensor, k: int = 25,
                     nonneg: bool = True) -> SparseCode:
    """Batched greedy non-negative gradient pursuit. h [N, d], atoms [T, d] unit-norm."""
    h = h.float()
    n = h.shape[0]
    idx = torch.full((n, k), -1, dtype=torch.long)
    coef = torch.zeros(n, k)
    resid = h.clone()
    active = torch.ones(n, dtype=torch.bool)
    for step in range(k):
        corr = resid @ atoms.T  # [N, T]
        if step:
            used = idx[:, :step]
            corr.scatter_(1, used.clamp(min=0), float("-inf"))
        best_val, best = (corr if nonneg else corr.abs()).max(1)
        active &= best_val > 1e-8
        if not active.any():
            break
        idx[active, step] = best[active]
        S = atoms[idx[:, : step + 1].clamp(min=0)]  # [N, s, d]
        mask = (idx[:, : step + 1] >= 0).float()
        # gradient of 0.5||h - c S||^2 wrt c on the support, exact line search
        g = torch.einsum("nsd,nd->ns", S, resid) * mask
        Sg = torch.einsum("ns,nsd->nd", g, S)
        alpha = (g * g).sum(1) / (Sg * Sg).sum(1).clamp(min=1e-12)
        c = coef[:, : step + 1] + alpha[:, None] * g
        if nonneg:
            c = c.clamp(min=0)
        c = c * mask
        coef[active, : step + 1] = c[active]
        recon = torch.einsum("ns,nsd->nd", coef[:, : step + 1], S)
        resid = h - recon
    S = atoms[idx.clamp(min=0)]
    recon = torch.einsum("nk,nkd->nd", coef * (idx >= 0).float(), S)
    idx[(coef == 0) & (idx >= 0)] = -1
    return SparseCode(idx, coef * (idx >= 0).float(), recon)


def variance_explained(h: torch.Tensor, recon: torch.Tensor) -> float:
    """1 - ||(h - recon) - mean||^2 / ||h - mean(h)||^2 over a batch (centered)."""
    hc = h - h.mean(0)
    rc = (h - recon) - (h - recon).mean(0)
    return float(1 - (rc**2).sum() / (hc**2).sum())


def span_projection(h: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Coordinates of h in an orthonormal basis of span(vectors): [N, rank]. The hypothesis-
    driven attribute-word subspace (representation (c) in T-042)."""
    q, r = torch.linalg.qr(vectors.T.float())
    keep = r.diagonal().abs() > 1e-6 * r.diagonal().abs().max()
    return h.float() @ q[:, keep]
