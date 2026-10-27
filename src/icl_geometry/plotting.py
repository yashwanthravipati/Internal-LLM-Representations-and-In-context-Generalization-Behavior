"""Shared matplotlib style for report figures (T-060).

Categorical colors use a fixed order and never cycle past what a chart can validate (at most
three series for scatter-type plots). Text stays in ink colors; marks are thin; grid is
recessive. Semantic roles (cued/uncued, conditions, representations, models) always get the
same color in every figure. Layers are plotted on normalized depth (0 = embeddings,
100 = last block output).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# Fixed color per semantic role, shared by every figure.
ROLE_COLORS = {
    # attributes (Exp 2)
    "cued": SERIES[0],
    "uncued": SERIES[1],
    # Exp 2 conditions
    "main": SERIES[0],
    "random_label": INK_MUTED,
    "explicit_rule": SERIES[2],
    "ambiguous": SERIES[3],
    # query types
    "congruent": SERIES[2],
    "incongruent": SERIES[1],
    # predictors (Exp 1)
    "llm": SERIES[0],
    "theory": SERIES[1],
    "prototype": SERIES[2],
    "exemplar": SERIES[3],
    # representations (H5); random subspaces are drawn as a gray band
    "full": SERIES[0],
    "jspace": SERIES[1],
    "attr_span": SERIES[2],
    "logit_span": SERIES[3],
    "pca_r": SERIES[4],
    "random_r": INK_MUTED,
    # models
    "Qwen/Qwen3.5-0.8B": SERIES[5],
    "Qwen/Qwen3.5-2B-Base": SERIES[6],
}


def depth(state_index, n_states: int):
    """Normalized depth 0-100 of a residual-stream state index (0 = embeddings,
    n_states - 1 = output of the last block)."""
    return 100 * np.asarray(state_index, dtype=float) / (n_states - 1)


def watermark(fig, text: str) -> None:
    """Stamp a figure that must never reach the report (smoke / placeholder data)."""
    fig.text(0.5, 0.5, text, ha="center", va="center", rotation=25, fontsize=34,
             color=SERIES[7], alpha=0.25, weight="bold", zorder=100)
    fig.text(0.99, 0.005, text, ha="right", va="bottom", fontsize=7, color=SERIES[7])


def banner(fig, text: str) -> None:
    """Small corner label (e.g. EXPLORATORY) for figures that may appear with a caveat."""
    fig.text(0.99, 0.995, text, ha="right", va="top", fontsize=7, color=INK_SECONDARY,
             weight="bold")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "font.size": 9,
            "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        }
    )


def save(fig, path: str | Path, dpi: int = 200, formats: tuple[str, ...] = ()) -> Path:
    """Save to `path`; `formats` adds siblings with other suffixes (e.g. ("pdf",))."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    for ext in formats:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
