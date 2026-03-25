"""FastAPI dependency injection callables."""
from fastapi import HTTPException
from app.config import settings, Settings
from app.ml import registry
from app.ml.registry import ModelNotFoundError, ModelNotLoadedError


def get_settings() -> Settings:
    return settings


def get_detector(model_name: str | None = None):
    name = model_name or settings.default_model_name
    try:
        return registry.get_loaded(name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))
