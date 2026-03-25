"""All Pydantic request/response schemas."""
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Detection schemas ─────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    modality: Literal["text", "email", "audio", "video"] = "text"
    content: str = Field(..., min_length=1, description="The message or email body to analyze.")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional: sender, subject, headers, etc.",
    )
    model: str | None = Field(None, description="Override the default model. E.g. 'tfidf_svm'.")


class DetectResponse(BaseModel):
    is_scam: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str
    explanation: list[str] | None = None


class BatchDetectRequest(BaseModel):
    items: list[DetectRequest] = Field(..., max_length=100)
    model: str | None = None


class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]


# ── Training schemas ──────────────────────────────────────────────────────────

class ColumnMapOverride(BaseModel):
    """Per-request override of the default dataset column mapping."""
    text_column: str = "text"
    label_column: str = "label"
    label_positive_value: str = "spam"


class TrainRequest(BaseModel):
    model_name: str = Field(..., description="Which model to train. E.g. 'tfidf_logreg'.")
    dataset_path: str = Field(..., description="Filename relative to DATASET_DIR.")
    dataset_format: Literal["csv", "json"] = "csv"
    column_map: ColumnMapOverride | None = Field(
        None,
        description="Override column names if your dataset differs from config defaults.",
    )
    hyperparams: dict[str, Any] | None = Field(
        None,
        description="Model-specific hyperparameters. E.g. {'C': 0.5, 'max_features': 10000}.",
    )


class TrainingMetricsOut(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None


class TrainResponse(BaseModel):
    job_id: str
    status: Literal["completed", "failed"]
    metrics: TrainingMetricsOut | None = None
    error: str | None = None
