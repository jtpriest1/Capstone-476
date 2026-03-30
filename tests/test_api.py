"""Integration tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.model import TrainingMetrics


client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_list():
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    models = r.json()["models"]
    names = [m["name"] for m in models]
    assert "tfidf_logreg" in names
    assert "transformer" in names


def test_detect_unloaded_model_returns_503():
    r = client.post("/api/v1/detect", json={"content": "You won a prize!"})
    # Model not trained yet → 503
    assert r.status_code == 503


def test_detect_with_loaded_model(monkeypatch):
    import app.model as model_mod

    mock_detector = MagicMock()
    mock_detector.name = "tfidf_logreg"
    mock_detector.is_loaded = True
    mock_detector.predict.return_value = (True, 0.92)
    mock_detector.explain.return_value = ["free", "won", "prize"]

    monkeypatch.setattr(model_mod, "_REGISTRY", {"tfidf_logreg": mock_detector})

    r = client.post("/api/v1/detect", json={"content": "You won a free prize!"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_scam"] is True
    assert body["confidence"] == pytest.approx(0.92, abs=0.01)
    assert "free" in body["explanation"]


def test_detect_image_no_text(monkeypatch):
    import io
    from PIL import Image
    from app import data as data_mod

    # Blank white image — OCR will find no text
    monkeypatch.setattr(data_mod, "extract_text_from_image", lambda b: "")

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(buf, format="PNG")
    buf.seek(0)

    r = client.post("/api/v1/detect/image", files={"file": ("blank.png", buf, "image/png")})
    assert r.status_code == 400
    assert r.json()["detail"] == "No text could be extracted from image"


def test_train_unknown_model_returns_error():
    r = client.post("/api/v1/train", json={
        "model_name": "does_not_exist",
        "dataset_path": "fake.csv",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "failed"
    assert "not registered" in body["error"]
