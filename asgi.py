"""Root ASGI application entry point for cloud deployments (Render, Vercel, Docker)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_BACKEND = _ROOT / "backend"

for p in [str(_BACKEND), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.interfaces.api.main import app, create_app
except ModuleNotFoundError:
    from interfaces.api.main import app, create_app

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("asgi:app", host="0.0.0.0", port=port)
