import torch

from diffusion_model.diffusion import linear_beta_schedule, sample
from diffusion_model.models.unet import UNet


def test_sample_runs():
    device = "cpu"
    model = UNet(
        [3, 8, 16],
        [16, 8, 3],
        t_emdim=8,
        class_emdim=8,
        pixel_dims_enc=8,
        use_mid_blocks=False,
    )
    _, betas = linear_beta_schedule(4, device=device)
    images = sample(
        model,
        betas,
        shape=(1, 3, 8, 8),
        num_steps=4,
        num_images=3,
        device=device,
        progress=False,
    )
    assert isinstance(images, list)
    assert images[-1].shape == (1, 3, 8, 8)
