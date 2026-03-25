from fastapi import APIRouter, Depends
from app.schemas.train import TrainRequest, TrainResponse
from app.dependencies import get_settings
from app.config import Settings
from app.training.trainer import run_training

router = APIRouter(tags=["training"])


@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest, settings: Settings = Depends(get_settings)):
    """Trigger a synchronous training run. Returns metrics when complete."""
    return run_training(request, settings)
