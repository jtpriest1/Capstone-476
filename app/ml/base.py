"""BaseDetector — the extensibility contract for all scam/phishing detection models.

Every detector (text, audio, video) must implement this ABC. The API layer
only ever calls these methods — it never imports a concrete class directly.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None


class BaseDetector(ABC):
    # Subclasses must set these as class attributes
    name: str          # unique registry key, e.g. "tfidf_logreg"
    modality: str      # "text", "audio", "video"

    @abstractmethod
    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics:
        """Train on a list of text samples and binary labels (1 = scam)."""
        ...

    @abstractmethod
    def predict(self, content: str) -> tuple[bool, float]:
        """Return (is_scam, confidence 0.0–1.0) for a single input."""
        ...

    @abstractmethod
    def predict_batch(self, contents: list[str]) -> list[tuple[bool, float]]:
        """Batch prediction — same semantics as predict(), more efficient."""
        ...

    @abstractmethod
    def explain(self, content: str) -> list[str]:
        """Return top contributing tokens/features for the prediction."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseDetector":
        """Load a previously saved model from disk."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is trained and ready to serve predictions."""
        return self._loaded

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._loaded = False
