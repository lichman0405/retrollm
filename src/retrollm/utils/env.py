"""Environment helpers.

The CLI enforces execution inside an isolated environment (conda or venv).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv(project_root: Path) -> None:
    """Load .env from the project root if present."""
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


def in_virtual_env() -> bool:
    """Return True if running inside venv/conda."""
    return bool(
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_PREFIX")
        or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    )


def ensure_virtual_env() -> None:
    """Fail fast if not running inside an isolated environment."""
    if in_virtual_env():
        return
    raise RuntimeError(
        "RetroLLM must be run inside a virtual environment (conda or venv). "
        "See environment.yml for a reproducible setup."
    )
