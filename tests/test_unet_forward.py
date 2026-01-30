import torch

from diffusion_model.models.unet import UNet


def test_unet_forward_shape():
    model = UNet(
        [3, 8, 16],
        [16, 8, 3],
        t_emdim=8,
        class_emdim=8,
        pixel_dims_enc=8,
        use_mid_blocks=False,
    )
    x = torch.randn(2, 3, 8, 8)
    t = torch.randint(0, 10, (2, 1)).float()
    out = model(x, t, None)
    assert out.shape == x.shape
