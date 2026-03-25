"""HuggingFace transformer-based detector — placeholder until audio/text deep-learning phase.

To activate:
  1. pip install torch transformers
  2. Implement train(), predict(), predict_batch() below.
  3. The @register decorator and API routing require no changes.
"""
from pathlib import Path
from app.ml.base import BaseDetector, TrainingMetrics
from app.ml.registry import register


@register
class TransformerDetector(BaseDetector):
    name = "transformer"
    modality = "text"

    # Intended model: distilbert-base-uncased (fast) or roberta-base (accurate)
    # Fine-tune on the scam dataset with a classification head.

    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics:
        raise NotImplementedError(
            "TransformerDetector is not yet implemented. "
            "Install torch + transformers and implement this method."
        )

    def predict(self, content: str) -> tuple[bool, float]:
        raise NotImplementedError("TransformerDetector is not yet implemented.")

    def predict_batch(self, contents: list[str]) -> list[tuple[bool, float]]:
        raise NotImplementedError("TransformerDetector is not yet implemented.")

    def explain(self, content: str) -> list[str]:
        raise NotImplementedError("TransformerDetector is not yet implemented.")

    def save(self, path: Path) -> None:
        raise NotImplementedError("TransformerDetector is not yet implemented.")

    @classmethod
    def load(cls, path: Path) -> "TransformerDetector":
        raise NotImplementedError("TransformerDetector is not yet implemented.")
