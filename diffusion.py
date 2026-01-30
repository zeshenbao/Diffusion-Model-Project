"""Legacy compatibility shim. Prefer `diffusion_model` package modules."""

from diffusion_model.diffusion import *  # noqa: F403
from diffusion_model.training import train  # noqa: F401


def main() -> None:
    from diffusion_model.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
