import torch

from diffusion_model.diffusion import cosine_schedule, linear_beta_schedule


def test_cosine_schedule_shapes():
    alphas, betas = cosine_schedule(10, device="cpu")
    assert alphas.shape == (10,)
    assert betas.shape == (10,)
    assert torch.all(alphas > 0)
    assert torch.all(betas > 0)


def test_linear_schedule_shapes():
    alphas, betas = linear_beta_schedule(5, device="cpu")
    assert alphas.shape == (5,)
    assert betas.shape == (5,)
