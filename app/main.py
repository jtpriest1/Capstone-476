"""FastAPI application factory."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ml import registry

# Import detector modules so @register decorators fire at startup
import app.ml.text.tfidf_classifier  # noqa: F401
import app.ml.text.transformer        # noqa: F401

from app.api.routes_health import router as health_router
from app.api.routes_detect import router as detect_router
from app.api.routes_train import router as train_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup: load any persisted models from disk
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    registry.try_load_from_disk(settings.model_dir)
    yield
    # On shutdown: nothing to clean up for now


def create_app() -> FastAPI:
    app = FastAPI(
        title="Scam Detection API",
        description="ML backend for detecting phishing/scam content in text, email, and (future) audio/video.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(health_router, prefix=prefix)
    app.include_router(detect_router, prefix=prefix)
    app.include_router(train_router, prefix=prefix)

    return app


app = create_app()
