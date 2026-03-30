"""FastAPI application — routes, dependencies, training orchestration, and app factory."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.config import ColumnMap, Settings, settings
from app.data import DatasetLoadError, clean_text, extract_email_parts, load_dataset
from app.model import (
    ModelNotFoundError,
    ModelNotLoadedError,
    get,
    get_loaded,
    list_models,
    try_load_from_disk,
)
from app.schemas import (
    BatchDetectRequest,
    BatchDetectResponse,
    DetectRequest,
    DetectResponse,
    TrainRequest,
    TrainResponse,
    TrainingMetricsOut,
)


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_settings() -> Settings:
    return settings


def get_detector(model_name: str | None = None):
    name = model_name or settings.default_model_name
    try:
        return get_loaded(name)
    except ModelNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(request: TrainRequest, cfg: Settings) -> TrainResponse:
    job_id = f"train_{request.model_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    if request.column_map:
        col_map = ColumnMap(
            text_column=request.column_map.text_column,
            label_column=request.column_map.label_column,
            label_positive_value=request.column_map.label_positive_value,
        )
    else:
        col_map = cfg.default_column_map

    try:
        detector = get(request.model_name)
    except ModelNotFoundError as e:
        return TrainResponse(job_id=job_id, status="failed", error=str(e))

    try:
        X, y = load_dataset(cfg.dataset_dir / request.dataset_path, request.dataset_format, col_map)
    except DatasetLoadError as e:
        return TrainResponse(job_id=job_id, status="failed", error=str(e))

    try:
        metrics = detector.train(X, y, request.hyperparams or {})
    except Exception as e:
        return TrainResponse(job_id=job_id, status="failed", error=f"Training failed: {e}")

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    detector.save(cfg.model_dir / f"{request.model_name}.joblib")

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


# ── Routes ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/api/v1")


@router.get("/health", tags=["health"])
def health():
    return {"status": "ok", "version": "0.1.0"}


@router.get("/models", tags=["health"])
def models_list():
    return {"models": list_models()}


def _run_detection(request: DetectRequest, cfg: Settings) -> DetectResponse:
    detector = get_detector(request.model or cfg.default_model_name)
    if request.modality == "email":
        content = extract_email_parts(request.content)["cleaned_body"]
    else:
        content = clean_text(request.content)
    is_scam, confidence = detector.predict(content)
    explanation = detector.explain(content)
    return DetectResponse(
        is_scam=is_scam,
        confidence=round(confidence, 4),
        model_used=detector.name,
        explanation=explanation or None,
    )


@router.post("/detect", response_model=DetectResponse, tags=["detection"])
def detect(request: DetectRequest, cfg: Settings = Depends(get_settings)):
    return _run_detection(request, cfg)


@router.post("/detect/batch", response_model=BatchDetectResponse, tags=["detection"])
def detect_batch(request: BatchDetectRequest, cfg: Settings = Depends(get_settings)):
    return BatchDetectResponse(results=[_run_detection(item, cfg) for item in request.items])


@router.post("/train", response_model=TrainResponse, tags=["training"])
def train(request: TrainRequest, cfg: Settings = Depends(get_settings)):
    """Trigger a synchronous training run. Returns metrics when complete."""
    return run_training(request, cfg)


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    try_load_from_disk(settings.model_dir)
    yield


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
    app.include_router(router)
    return app


app = create_app()
