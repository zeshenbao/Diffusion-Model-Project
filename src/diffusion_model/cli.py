"""Command line interface for training and sampling."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from diffusion_model.data import create_dataloader
from diffusion_model.diffusion import linear_beta_schedule, sample
from diffusion_model.models.unet import UNet
from diffusion_model.training import load_checkpoint, save_checkpoint, train
from diffusion_model.utils import resolve_device, set_seed


def build_model(
    *,
    channels: int,
    img_size: int,
    t_embed_dim: int = 32,
    class_embed_dim: int = 32,
    use_cross_attn: bool = False,
    use_mid_blocks: bool = True,
    fast_attn: bool = False,
) -> UNet:
    enc_channels = [channels, 64, 128, 256, 512]
    dec_channels = [512, 256, 128, 64, channels]
    return UNet(
        enc_channels,
        dec_channels,
        t_embed_dim,
        class_emdim=class_embed_dim,
        pixel_dims_enc=img_size,
        use_cross_attn=use_cross_attn,
        use_mid_blocks=use_mid_blocks,
        fast_attn=fast_attn,
    )


def train_cmd(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(args.seed)

    channels = 1 if args.dataset == "mnist" else 3
    model = build_model(
        channels=channels,
        img_size=args.img_size,
        t_embed_dim=args.t_embed_dim,
        class_embed_dim=args.class_embed_dim,
        use_cross_attn=args.use_cross_attn,
        use_mid_blocks=args.use_mid_blocks,
        fast_attn=args.fast_attn,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    dataloader = create_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_dir=args.data_dir,
        download=not args.no_download,
        num_workers=args.num_workers,
    )

    total_steps = max(1, len(dataloader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)

    alphas, betas = linear_beta_schedule(
        args.time_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    losses = []
    start_epoch = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        losses = state.losses
        start_epoch = state.epoch

    ema_model = deepcopy(model) if args.use_ema else None

    losses, ema_model = train(
        dataloader,
        betas,
        model,
        optimizer,
        epochs=args.epochs,
        alphas=alphas,
        device=device,
        ema_model=ema_model,
        clip_grad=args.clip_grad,
        use_class_cond=not args.disable_class_cond,
        scheduler=scheduler,
        start_epoch=start_epoch,
        losses=losses,
        save_interval=args.save_interval,
        save_path=args.save_path,
    )

    if args.save_path:
        save_checkpoint(
            args.save_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=start_epoch + args.epochs,
            losses=losses,
            save_interval=args.save_interval,
            ema_model=ema_model,
        )


@torch.no_grad()
def sample_cmd(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    channels = args.channels
    model = build_model(
        channels=channels,
        img_size=args.img_size,
        t_embed_dim=args.t_embed_dim,
        class_embed_dim=args.class_embed_dim,
        use_cross_attn=args.use_cross_attn,
        use_mid_blocks=args.use_mid_blocks,
        fast_attn=args.fast_attn,
    ).to(device)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, map_location=device)

    _, betas = linear_beta_schedule(
        args.time_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    class_labels = None
    if args.class_labels:
        class_labels = [int(x) for x in args.class_labels.split(",")]

    images = sample(
        model,
        betas,
        shape=(args.num_samples, channels, args.img_size, args.img_size),
        num_steps=args.time_steps,
        num_images=args.num_images,
        class_labels=class_labels,
        guidance_scale=args.guidance_scale,
        device=device,
        progress=not args.no_progress,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(images, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diffusion Model CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the diffusion model")
    train_parser.add_argument("--dataset", default="cifar", choices=["mnist", "cifar", "lfw"])
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--no-download", action="store_true")
    train_parser.add_argument("--batch-size", type=int, default=24)
    train_parser.add_argument("--img-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--clip-grad", type=float, default=1.0)
    train_parser.add_argument("--time-steps", type=int, default=1000)
    train_parser.add_argument("--beta-start", type=float, default=1e-4)
    train_parser.add_argument("--beta-end", type=float, default=2e-2)
    train_parser.add_argument("--eta-min", type=float, default=1e-7)
    train_parser.add_argument("--t-embed-dim", type=int, default=32)
    train_parser.add_argument("--class-embed-dim", type=int, default=32)
    train_parser.add_argument("--use-cross-attn", action="store_true")
    train_parser.add_argument("--no-mid-blocks", action="store_false", dest="use_mid_blocks", default=True)
    train_parser.add_argument("--fast-attn", action="store_true")
    train_parser.add_argument("--disable-class-cond", action="store_true")
    train_parser.add_argument("--use-ema", action="store_true")
    train_parser.add_argument("--save-interval", type=int, default=50)
    train_parser.add_argument("--save-path", type=str, default="checkpoints/model.pth")
    train_parser.add_argument("--resume", type=str)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--device", type=str)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train_cmd)

    sample_parser = subparsers.add_parser("sample", help="Sample images from a trained model")
    sample_parser.add_argument("--checkpoint", type=str, required=True)
    sample_parser.add_argument("--output", type=str, default="outputs/samples.pt")
    sample_parser.add_argument("--num-samples", type=int, default=10)
    sample_parser.add_argument("--num-images", type=int, default=10)
    sample_parser.add_argument("--img-size", type=int, default=32)
    sample_parser.add_argument("--channels", type=int, default=3)
    sample_parser.add_argument("--time-steps", type=int, default=1000)
    sample_parser.add_argument("--beta-start", type=float, default=1e-4)
    sample_parser.add_argument("--beta-end", type=float, default=2e-2)
    sample_parser.add_argument("--guidance-scale", type=float, default=0.0)
    sample_parser.add_argument("--class-labels", type=str)
    sample_parser.add_argument("--t-embed-dim", type=int, default=32)
    sample_parser.add_argument("--class-embed-dim", type=int, default=32)
    sample_parser.add_argument("--use-cross-attn", action="store_true")
    sample_parser.add_argument("--no-mid-blocks", action="store_false", dest="use_mid_blocks", default=True)
    sample_parser.add_argument("--fast-attn", action="store_true")
    sample_parser.add_argument("--no-progress", action="store_true")
    sample_parser.add_argument("--device", type=str)
    sample_parser.set_defaults(func=sample_cmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
