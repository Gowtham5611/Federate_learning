"""Main entry point for FedSecureAI FastAPI inference and evaluation service.

Allows starting the service with:
  uvicorn backend.interfaces.api.main:app --host 0.0.0.0 --port 8000
or:
  python -m backend.interfaces.api.main
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend and repository root are on sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_ROOT_DIR = _BACKEND_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from interfaces.api.app import app, create_app
except ModuleNotFoundError:
    from backend.interfaces.api.app import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.interfaces.api.main:app", host="0.0.0.0", port=8000, reload=True)

