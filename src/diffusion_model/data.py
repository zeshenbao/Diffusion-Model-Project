"""Dataset utilities."""

from __future__ import annotations

from typing import Literal

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar", "lfw"]


def create_dataloader(
    dataset: DatasetName,
    *,
    batch_size: int,
    img_size: int,
    data_dir: str = "data",
    download: bool = True,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Create a PyTorch dataloader for supported datasets."""
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    if dataset == "mnist":
        ds = datasets.MNIST(root=f"{data_dir}/mnist", train=True, transform=transform, download=download)
    elif dataset == "cifar":
        ds = datasets.CIFAR10(root=f"{data_dir}/cifar", train=True, transform=transform, download=download)
    elif dataset == "lfw":
        ds = datasets.LFWPeople(root=f"{data_dir}/lfw", split="train", transform=transform, download=download)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
