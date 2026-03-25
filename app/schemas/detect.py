from typing import Any, Literal
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    modality: Literal["text", "email", "audio", "video"] = "text"
    content: str = Field(..., min_length=1, description="The message or email body to analyze.")
    metadata: dict[str, Any] | None = Field(
        None,
        description="Optional: sender, subject, headers, etc. Not used by models yet but stored for future use.",
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
