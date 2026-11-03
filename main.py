"""Root entry point that keeps the historical ``python main.py ...`` commands.

The real composition root lives at ``backend/composition_root.py``. This module
is deliberately thin: it puts ``backend/`` on ``sys.path`` and delegates, so the
commands that shipped with the project - ``python main.py dataset-info``,
``python main.py dataset-download`` - keep working unchanged. Nothing beyond
that lives here; no adapters are constructed and no configuration is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The composition_root module imports the project packages as top-level modules
# (``from application.services.dataset_manager import DatasetManager``), so the
# backend source tree must be importable regardless of how this file is invoked.
_BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from composition_root import main  # noqa: E402  (import after sys.path setup)

if __name__ == "__main__":
    raise SystemExit(main())
