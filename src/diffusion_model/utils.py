"""Utility helpers for the diffusion project."""

from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a torch device, defaulting to CUDA when available."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
