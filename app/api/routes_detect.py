from fastapi import APIRouter, Depends
from app.schemas.detect import DetectRequest, DetectResponse, BatchDetectRequest, BatchDetectResponse
from app.dependencies import get_detector, get_settings
from app.config import Settings
from app.data.preprocessor import clean_text, extract_email_parts

router = APIRouter(tags=["detection"])


def _run_detection(request: DetectRequest, settings: Settings) -> DetectResponse:
    model_name = request.model or settings.default_model_name
    detector = get_detector(model_name)

    if request.modality == "email":
        parts = extract_email_parts(request.content)
        content = parts["cleaned_body"]
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


@router.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest, settings: Settings = Depends(get_settings)):
    return _run_detection(request, settings)


@router.post("/detect/batch", response_model=BatchDetectResponse)
def detect_batch(request: BatchDetectRequest, settings: Settings = Depends(get_settings)):
    results = [_run_detection(item, settings) for item in request.items]
    return BatchDetectResponse(results=results)
