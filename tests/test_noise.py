import torch

from diffusion_model.diffusion import apply_noise


def test_apply_noise_shapes():
    x = torch.zeros(3, 8, 8)
    alpha = torch.linspace(0.1, 0.9, 4)
    noisy_x, eps = apply_noise(x, alpha)
    assert noisy_x.shape == (4, 3, 8, 8)
    assert eps.shape == (4, 3, 8, 8)
