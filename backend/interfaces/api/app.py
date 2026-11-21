"""FastAPI application factory and lifecycle management for FedSecureAI."""

from __future__ import annotations

import gc
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# Configure bounded threading for cloud environments (prevents memory spikes on 512MB instances)
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

# Ensure backend and repository root are always on sys.path
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_ROOT_DIR = _BACKEND_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette import status

try:
    from interfaces.api.middleware.request_guards import RequestTimingAndPrivacyMiddleware
    from interfaces.api.routes.health import router as health_router
    from interfaces.api.routes.inference import router as inference_router
    from interfaces.api.routes.evaluation import router as evaluation_router
    from interfaces.api.routes.research import router as research_router
    from interfaces.api.services.model_service import ModelService
except ModuleNotFoundError:
    from backend.interfaces.api.middleware.request_guards import RequestTimingAndPrivacyMiddleware
    from backend.interfaces.api.routes.health import router as health_router
    from backend.interfaces.api.routes.inference import router as inference_router
    from backend.interfaces.api.routes.evaluation import router as evaluation_router
    from backend.interfaces.api.routes.research import router as research_router
    from backend.interfaces.api.services.model_service import ModelService

logger = logging.getLogger(__name__)


def create_app(preload_models: bool = True) -> FastAPI:
    """Create and configure a FedSecureAI FastAPI application instance.

    Args:
        preload_models: If True, load and verify model weights during startup lifespan.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        logger.info("Initializing FedSecureAI FastAPI inference service...")
        if preload_models:
            try:
                model_service = ModelService()
                model_service.load_all_models()
                gc.collect()
                logger.info("All federated AML models successfully loaded and verified.")
            except Exception as e:
                logger.error("Failed to load models during startup: %s", e)
        yield
        # Shutdown
        logger.info("Shutting down FedSecureAI FastAPI inference service...")

    app = FastAPI(
        title="FedSecureAI Inference Service",
        description="Privacy-Preserving, Explainable, and Drift-Aware Federated AML Detection API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Privacy and timing middleware
    app.add_middleware(RequestTimingAndPrivacyMiddleware)

    # 2. CORS middleware (supporting local interfaces and Vercel cloud deployments)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://federate-learning.vercel.app",
        ],
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Sanitized Global Exception Handler
    @app.exception_handler(Exception)
    async def global_sanitized_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception processing %s: %s", request.url.path, exc, exc_info=True)
        # Never leak internal file paths, stack traces, or credentials to client
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred. Transaction trace logged securely."},
        )

    # 4. Route registration (Root, /v1, /api/v1, and /api aliases for client compatibility)
    for prefix in ("", "/v1", "/api/v1", "/api"):
        app.include_router(health_router, prefix=prefix)
        app.include_router(inference_router, prefix=prefix)
        app.include_router(evaluation_router, prefix=prefix)
        app.include_router(research_router, prefix=prefix)

    return app


# Default app instance for ASGI servers (preloads models by default)
preload_default = os.getenv("FEDSECUREAI_PRELOAD_MODELS", "true").lower() in ("true", "1")
app = create_app(preload_models=preload_default)
