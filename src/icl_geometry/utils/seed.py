"""Seeding utility for python, numpy and torch."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed python, numpy (global) and torch; return a fresh numpy Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)
