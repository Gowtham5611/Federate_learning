"""Project path resolution.

Single source of truth for the project root so that no other module has to
count ``parents[n]`` levels or depend on the current working directory.

Two roots matter, and they are not the same directory:

* :data:`PROJECT_ROOT` is the repository root. Shared, non-code areas live
  there - ``data/``, ``datasets/``, ``reports/``, ``docs/``, ``logs/``, ``.env``
  - and they are shared with the frontend, so they must never be resolved
  relative to the backend.
* :data:`BACKEND_ROOT` is the backend source tree, which owns the backend's own
  configuration.
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives at <root>/backend/infrastructure/utils/paths.py.
#   parents[0] utils  parents[1] infrastructure  parents[2] backend  parents[3] root
BACKEND_ROOT: Path = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Path = BACKEND_ROOT.parent

CONFIG_DIR: Path = BACKEND_ROOT / "config"
DATASETS_DIR: Path = PROJECT_ROOT / "datasets"


def resolve(relative_path: str | Path) -> Path:
    """Resolve a project-relative path against :data:`PROJECT_ROOT`.

    Absolute paths are returned unchanged, so configuration values may use
    either form.
    """
    path = Path(relative_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
