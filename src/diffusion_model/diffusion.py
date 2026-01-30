"""Core diffusion utilities (schedules, noise, sampling, EMA)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn
from tqdm import tqdm

from diffusion_model.utils import resolve_device


def calc_alphas(betas: torch.Tensor) -> torch.Tensor:
    """Compute cumulative product of (1 - beta)."""
    return torch.cumprod(torch.ones_like(betas) - betas, dim=0)


def cosine_schedule(
    num_steps: int,
    *,
    s: float = 0.008,
    max_beta: float = 0.999,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine schedule from https://arxiv.org/abs/2102.09672.

    Returns:
        alphas: cumulative product schedule (length num_steps).
        betas: per-step noise values (length num_steps).
    """
    device = resolve_device(device)
    steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
    angles = (steps / num_steps + s) / (1 + s) * math.pi / 2
    alphas_bar = torch.cos(angles) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    betas = betas.clamp(max=max_beta)
    alphas = calc_alphas(betas)
    return alphas, betas


def linear_beta_schedule(
    num_steps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear beta schedule."""
    device = resolve_device(device)
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = calc_alphas(betas)
    return alphas, betas


def apply_noise(x: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply noise to a single sample across multiple alpha values.

    Args:
        x: Tensor of shape (C,H,W) or (H,W).
        alpha: Tensor of shape (T,) with cumulative alphas.

    Returns:
        noisy_x: Tensor of shape (T,C,H,W) or (T,H,W).
        eps: Noise tensor with the same shape as noisy_x.
    """
    if x.dim() not in (2, 3):
        raise ValueError("apply_noise expects a single sample with 2D or 3D shape")

    x = x.unsqueeze(0)
    alpha = alpha.to(x.device)
    x_exp = x.expand(alpha.shape[0], *x.shape[1:])
    eps = torch.randn_like(x_exp)

    alpha_view = alpha.view(-1, *([1] * (x_exp.dim() - 1)))
    noisy_x = torch.sqrt(alpha_view) * x_exp + torch.sqrt(1 - alpha_view) * eps
    return noisy_x, eps


def ema_param_update(
    model: nn.Module,
    ema_copy: nn.Module | None = None,
    *,
    decay: float = 0.9999,
) -> nn.Module:
    """EMA update of the main model."""
    if ema_copy is None:
        ema_copy = deepcopy(model).eval()
        for param in ema_copy.parameters():
            param.requires_grad_(False)
    else:
        ema_copy.eval()

    for ema_param, param in zip(ema_copy.parameters(), model.parameters(), strict=False):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    return ema_copy


@torch.no_grad()
def sample(
    model: nn.Module,
    betas: torch.Tensor,
    *,
    shape: tuple[int, int, int, int],
    num_steps: int | None = None,
    num_images: int = 10,
    start_img: torch.Tensor | None = None,
    alphas: torch.Tensor | None = None,
    class_labels: Iterable[int] | None = None,
    guidance_scale: float = 0.0,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> list[torch.Tensor]:
    """Generate samples from a trained diffusion model.

    Args:
        model: Trained model.
        betas: Beta schedule of shape (T,).
        shape: Shape of generated images (B,C,H,W).
        num_steps: Number of diffusion steps. Defaults to betas length.
        num_images: Number of intermediate images to save.
        start_img: Optional starting image (for partial diffusion).
        alphas: Optional precomputed cumulative alphas.
        class_labels: Optional class labels for conditional generation.
        guidance_scale: Classifier-free guidance scale.
        device: Device to run on.
        progress: Whether to show a progress bar.
    """
    device = resolve_device(device)
    num_steps = num_steps or betas.shape[0]
    betas = betas.to(device)

    if alphas is None:
        alphas = calc_alphas(betas)
    else:
        alphas = alphas.to(device)

    x = torch.randn(size=shape, device=device) if start_img is None else start_img.to(device)

    if class_labels is not None:
        class_labels = torch.tensor(
            list(class_labels),
            dtype=torch.int64,
            device=device,
        ).view(-1, 1)

    if start_img is not None:
        t_start = torch.full((x.shape[0], 1), num_steps - 1, device=device)
        eps = torch.randn_like(x)
        alpha_t = alphas[t_start]
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        x = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * eps

    sigmas = torch.sqrt(betas)

    if num_images <= 1:
        save_steps = {0}
    else:
        save_steps = set(torch.linspace(0, num_steps - 1, num_images, dtype=torch.long).tolist())

    saved_images: list[torch.Tensor] = []
    model.eval()

    iterator = range(num_steps - 1, -1, -1)
    if progress:
        iterator = tqdm(iterator)

    for t in iterator:
        noise = torch.randn(size=shape, device=device) if t >= 1 else torch.zeros_like(x)
        t_step = torch.full((x.shape[0], 1), float(t), device=device)

        model_output = model(x.float(), t_step, class_labels)
        if guidance_scale > 0 and class_labels is not None:
            uncond = model(x.float(), t_step, None)
            model_output = uncond + guidance_scale * (model_output - uncond)

        x = x - betas[t] * model_output / torch.sqrt(1 - alphas[t])
        x = x / torch.sqrt(1 - betas[t]) + sigmas[t] * noise

        if t in save_steps:
            saved_images.append(x.detach().cpu())

    if not saved_images:
        saved_images.append(x.detach().cpu())
    return saved_images
