from typing import Any, Literal
from pydantic import BaseModel, Field


class ColumnMapOverride(BaseModel):
    """Per-request override of the default dataset column mapping."""
    text_column: str = "text"
    label_column: str = "label"
    label_positive_value: str = "spam"


class TrainRequest(BaseModel):
    model_name: str = Field(..., description="Which model to train. E.g. 'tfidf_logreg'.")
    dataset_path: str = Field(..., description="Filename relative to DATASET_DIR. E.g. 'sms_spam.csv'.")
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
