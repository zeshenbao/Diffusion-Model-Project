"""Training loop and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model.diffusion import calc_alphas, ema_param_update
from diffusion_model.utils import resolve_device


@dataclass
class TrainState:
    epoch: int
    losses: list[float]
    save_interval: Optional[int] = None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    losses: Optional[list[float]] = None,
    save_interval: Optional[int] = None,
    ema_model: Optional[nn.Module] = None,
) -> None:
    """Save model (and optional optimizer/scheduler) to disk."""
    path = Path(path)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss": losses or [],
        "save_interval": save_interval,
    }
    torch.save(state, path)

    if ema_model is not None:
        ema_state = {
            "epoch": epoch,
            "model_state_dict": ema_model.state_dict(),
            "loss": losses or [],
            "save_interval": save_interval,
        }
        torch.save(ema_state, path.with_name(f"EMA_{path.name}"))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: Optional[str | torch.device] = "cpu",
) -> TrainState:
    """Load model (and optional optimizer/scheduler) from disk."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return TrainState(
        epoch=checkpoint.get("epoch", 0),
        losses=checkpoint.get("loss", []),
        save_interval=checkpoint.get("save_interval"),
    )


def train(
    dataloader: DataLoader,
    betas: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 10,
    alphas: Optional[torch.Tensor] = None,
    device: Optional[str | torch.device] = None,
    ema_model: Optional[nn.Module] = None,
    clip_grad: float = 1.0,
    use_class_cond: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    start_epoch: int = 0,
    losses: Optional[list[float]] = None,
    save_interval: Optional[int] = None,
    save_path: Optional[str | Path] = None,
) -> tuple[list[float], Optional[nn.Module]]:
    """Train the diffusion model for the given number of epochs."""
    device = resolve_device(device)
    model = model.to(device)
    model.train()

    if alphas is None:
        alphas = calc_alphas(betas)
    alphas = alphas.to(device)
    betas = betas.to(device)

    losses = losses or []
    supervised_loss = nn.MSELoss().to(device)

    for epoch in range(epochs):
        step_count = 0
        for x, y in tqdm(dataloader):
            step_count += 1
            if use_class_cond:
                y = y.reshape(-1, 1).to(device)
                if torch.rand(1).item() < 0.1:
                    y = None
            else:
                y = None

            x = x.to(device)
            t = torch.randint(0, alphas.shape[0], (x.shape[0], 1), device=device)
            eps = torch.randn_like(x)

            alpha_t = torch.sqrt(alphas[t]).view(-1, 1, 1, 1)
            one_minus_alpha = torch.sqrt(1.0 - alphas[t]).view(-1, 1, 1, 1)
            noisy_x = alpha_t * x + one_minus_alpha * eps

            model_output = model(noisy_x.float(), t.float(), y)
            loss = supervised_loss(model_output, eps)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            losses.append(loss.item())
            if ema_model is not None:
                ema_model = ema_param_update(model, ema_model)

            if save_interval and save_path and (step_count % save_interval == 0):
                save_checkpoint(
                    save_path,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + start_epoch,
                    losses=losses,
                    save_interval=save_interval,
                    ema_model=ema_model,
                )

        if save_path:
            save_checkpoint(
                save_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + start_epoch + 1,
                losses=losses,
                save_interval=save_interval,
                ema_model=ema_model,
            )

    return losses, ema_model
