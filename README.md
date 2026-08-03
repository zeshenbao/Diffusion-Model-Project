# Diffusion Model Project

Denoising Diffusion Probabilistic Model (DDPM) with a U-Net backbone and attention.

## Project Report

Read the full 12-page technical report, [*Pic-AI-sso, Because More is More*](docs/project-report.pdf), for the model design, training methodology, experiments, evaluation, and results.

## Highlights

- Clean `src/` package layout with reusable modules.
- U-Net with attention blocks and optional cross-attention.
- Training with EMA, gradient clipping, and cosine LR scheduling.
- Sampling with optional classifier-free guidance.
- Pytest suite and GitHub Actions CI.

## Project Layout

```
.
├── src/
│   └── diffusion_model/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── diffusion.py
│       ├── training.py
│       ├── utils.py
│       └── models/
│           └── unet.py
├── tests/
├── docs/
│   └── project-report.pdf
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Train

```bash
python -m diffusion_model.cli train \
  --dataset cifar \
  --epochs 10 \
  --batch-size 64 \
  --save-path checkpoints/model.pth
```

## Sample

```bash
python -m diffusion_model.cli sample \
  --checkpoint checkpoints/model.pth \
  --output outputs/samples.pt \
  --num-samples 16
```

## Tests & Linting

```bash
ruff check .
ruff format --check .
pytest
```

## CI/CD

GitHub Actions workflow in `.github/workflows/ci.yml` runs linting and tests on every push and pull request.

## Notes

- Datasets are downloaded into `data/` by default.
- For MNIST, use `--dataset mnist` (1-channel). For CIFAR, use `--dataset cifar` (3-channel).
- Sample outputs are saved as PyTorch tensors; visualize with `diffusion_model.visualization.montage`.

---

Originally developed for DD2424 at KTH by Adam Fredriksson, Astrid Frykman, Jacob Westergren, and Zeshen Bao.
