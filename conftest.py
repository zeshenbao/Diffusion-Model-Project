"""Pytest configuration to make the src layout importable without installation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `import diffusion_model` works when running tests from the repo root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.is_dir():
    src_path = str(SRC)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
