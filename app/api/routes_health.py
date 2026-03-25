from fastapi import APIRouter
from app.ml import registry

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/models")
def list_models():
    return {"models": registry.list_models()}
