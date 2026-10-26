import torch
import torch.nn.functional as F

from icl_geometry.models.fast_kernels import depthwise_causal_conv1d


def reference(x, w, b, activation=None):
    out = F.conv1d(x, w.unsqueeze(1), b, padding=w.shape[-1] - 1, groups=x.shape[1])
    out = out[:, :, : x.shape[-1]]
    return F.silu(out) if activation == "silu" else out


def test_matches_depthwise_conv():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 16, 11, generator=g)
    w = torch.randn(16, 4, generator=g)
    b = torch.randn(16, generator=g)
    for act in (None, "silu"):
        torch.testing.assert_close(
            depthwise_causal_conv1d(x, w, b, activation=act), reference(x, w, b, act)
        )
    torch.testing.assert_close(depthwise_causal_conv1d(x, w, None), reference(x, w, None))
