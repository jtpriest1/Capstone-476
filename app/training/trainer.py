"""Training orchestrator — loads dataset, preprocesses, trains, persists, returns metrics."""
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, ColumnMap
from app.data.loader import load_dataset, DatasetLoadError
from app.ml import registry
from app.ml.registry import ModelNotFoundError
from app.schemas.train import TrainRequest, TrainResponse, TrainingMetricsOut, ColumnMapOverride


def run_training(request: TrainRequest, settings: Settings) -> TrainResponse:
    job_id = f"train_{request.model_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    # Resolve column map: request override > config defaults
    if request.column_map:
        col_map = ColumnMap(
            text_column=request.column_map.text_column,
            label_column=request.column_map.label_column,
            label_positive_value=request.column_map.label_positive_value,
        )
    else:
        col_map = settings.default_column_map

    dataset_path = settings.dataset_dir / request.dataset_path

    try:
        detector = registry.get(request.model_name)
    except ModelNotFoundError as e:
        return TrainResponse(job_id=job_id, status="failed", error=str(e))

    try:
        X, y = load_dataset(dataset_path, request.dataset_format, col_map)
    except DatasetLoadError as e:
        return TrainResponse(job_id=job_id, status="failed", error=str(e))

    try:
        hyperparams = request.hyperparams or {}
        metrics = detector.train(X, y, hyperparams)
    except Exception as e:
        return TrainResponse(job_id=job_id, status="failed", error=f"Training failed: {e}")

    model_path = settings.model_dir / f"{request.model_name}.joblib"
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    detector.save(model_path)

    return TrainResponse(
        job_id=job_id,
        status="completed",
        metrics=TrainingMetricsOut(
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            auc_roc=metrics.auc_roc,
        ),
    )
