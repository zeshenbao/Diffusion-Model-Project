"""Visualization helpers."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def montage(images: Iterable[torch.Tensor], is_batched: bool = False) -> None:
    """Render a grid of images stored as tensors.

    Args:
        images: Iterable of images (C,H,W) or (B,C,H,W) tensors.
        is_batched: If True, expects images to have a batch dimension.
    """
    images = list(images)
    if not images:
        return

    num_plots = len(images)
    rows = 4 if num_plots % 4 == 0 else 2
    cols = max(1, num_plots // rows)
    fig, axes = plt.subplots(rows, cols, squeeze=False)

    for idx, (i, j) in enumerate([(i, j) for i in range(rows) for j in range(cols)]):
        if idx >= num_plots:
            axes[i][j].axis("off")
            continue
        temp = images[idx].detach().cpu().numpy()
        if is_batched or temp.ndim == 4:
            temp = temp[0]
        img = temp.transpose(1, 2, 0)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        axes[i][j].imshow(img)
        axes[i][j].axis("off")

    fig.tight_layout()
    plt.show()
