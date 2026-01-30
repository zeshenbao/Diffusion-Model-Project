"""Configuration dataclasses for training and sampling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    enc_channels: list[int] | None = None
    dec_channels: list[int] | None = None
    t_embed_dim: int = 32
    class_embed_dim: int = 32
    pixel_dims: int = 32
    use_cross_attn: bool = False
    use_mid_blocks: bool = True
    fast_attn: bool = False

    def __post_init__(self) -> None:
        if self.enc_channels is None:
            self.enc_channels = [3, 64, 128, 256, 512]
        if self.dec_channels is None:
            self.dec_channels = [512, 256, 128, 64, 3]


@dataclass
class DiffusionConfig:
    time_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "linear"


@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 24
    learning_rate: float = 3e-4
    clip_grad: float = 1.0
    save_interval: int = 50


@dataclass
class DataConfig:
    dataset: str = "cifar"
    img_size: int = 32
    data_dir: str = "data"
    num_workers: int = 0
