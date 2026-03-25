"""All ML components: base contract, registry, and concrete detectors."""
from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ── Base contract ────────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None = None


class BaseDetector(ABC):
    name: str      # unique registry key, e.g. "tfidf_logreg"
    modality: str  # "text", "audio", "video"

    @abstractmethod
    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics: ...

    @abstractmethod
    def predict(self, content: str) -> tuple[bool, float]: ...

    @abstractmethod
    def predict_batch(self, contents: list[str]) -> list[tuple[bool, float]]: ...

    @abstractmethod
    def explain(self, content: str) -> list[str]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseDetector": ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._loaded = False


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, BaseDetector] = {}


class ModelNotFoundError(Exception):
    pass


class ModelNotLoadedError(Exception):
    pass


def register(cls: type[BaseDetector]) -> type[BaseDetector]:
    """Class decorator — registers an instance of the detector by name."""
    _REGISTRY[cls.name] = cls()
    return cls


def get(name: str) -> BaseDetector:
    if name not in _REGISTRY:
        raise ModelNotFoundError(
            f"Model '{name}' is not registered. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def get_loaded(name: str) -> BaseDetector:
    detector = get(name)
    if not detector.is_loaded:
        raise ModelNotLoadedError(
            f"Model '{name}' is registered but not loaded. Train it first via POST /train."
        )
    return detector


def list_models() -> list[dict]:
    return [
        {"name": name, "loaded": d.is_loaded, "modality": d.modality}
        for name, d in _REGISTRY.items()
    ]


def try_load_from_disk(model_dir: Path) -> None:
    """On startup, load any persisted .joblib files found in model_dir."""
    for name, detector in _REGISTRY.items():
        model_path = model_dir / f"{name}.joblib"
        if model_path.exists():
            try:
                _REGISTRY[name] = detector.__class__.load(model_path)
                print(f"[registry] Loaded '{name}' from {model_path}")
            except Exception as e:
                print(f"[registry] Failed to load '{name}': {e}")


# ── TF-IDF classifiers ────────────────────────────────────────────────────────

class TfidfClassifier(BaseDetector):
    """Base for TF-IDF pipeline detectors. Subclass: set `name` and `_make_clf()`."""

    modality = "text"
    _pipeline: Pipeline | None = None

    def _make_clf(self, hyperparams: dict):
        raise NotImplementedError

    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=hyperparams.get("max_features", 20_000),
                ngram_range=tuple(hyperparams.get("ngram_range", [1, 2])),
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", self._make_clf(hyperparams)),
        ])
        self._pipeline.fit(X_train, y_train)
        self._loaded = True

        y_pred = self._pipeline.predict(X_val)
        y_proba = self._get_proba(X_val)
        return TrainingMetrics(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, zero_division=0),
            recall=recall_score(y_val, y_pred, zero_division=0),
            f1=f1_score(y_val, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_val, y_proba) if y_proba is not None else None,
        )

    def predict(self, content: str) -> tuple[bool, float]:
        self._assert_loaded()
        proba = self._get_proba([content])
        confidence = float(proba[0]) if proba is not None else self._binary_confidence([content])[0]
        return confidence >= 0.5, confidence

    def predict_batch(self, contents: list[str]) -> list[tuple[bool, float]]:
        self._assert_loaded()
        proba = self._get_proba(contents)
        if proba is not None:
            return [(bool(p >= 0.5), float(p)) for p in proba]
        return [(bool(c >= 0.5), c) for c in self._binary_confidence(contents)]

    def explain(self, content: str) -> list[str]:
        self._assert_loaded()
        tfidf: TfidfVectorizer = self._pipeline.named_steps["tfidf"]
        vec = tfidf.transform([content])
        feature_names = np.array(tfidf.get_feature_names_out())
        top_indices = np.argsort(vec.toarray()[0])[::-1][:10]
        return [feature_names[i] for i in top_indices if vec[0, i] > 0]

    def save(self, path: Path) -> None:
        joblib.dump({"pipeline": self._pipeline, "loaded": self._loaded}, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfClassifier":
        data = joblib.load(path)
        instance = cls()
        instance._pipeline = data["pipeline"]
        instance._loaded = data["loaded"]
        return instance

    def _get_proba(self, X: list[str]) -> np.ndarray | None:
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self._pipeline.predict_proba(X)[:, 1]
        return None

    def _binary_confidence(self, X: list[str]) -> list[float]:
        return [float(p) for p in self._pipeline.predict(X)]

    def _assert_loaded(self):
        if not self._loaded or self._pipeline is None:
            raise RuntimeError(f"Model '{self.name}' is not trained. Call train() first.")


@register
class TfidfLogReg(TfidfClassifier):
    name = "tfidf_logreg"

    def _make_clf(self, hyperparams: dict):
        return LogisticRegression(
            C=hyperparams.get("C", 1.0),
            max_iter=hyperparams.get("max_iter", 1000),
            class_weight="balanced",
        )


@register
class TfidfSVM(TfidfClassifier):
    name = "tfidf_svm"

    def _make_clf(self, hyperparams: dict):
        return CalibratedClassifierCV(
            LinearSVC(
                C=hyperparams.get("C", 1.0),
                max_iter=hyperparams.get("max_iter", 2000),
                class_weight="balanced",
            )
        )


@register
class TfidfRandomForest(TfidfClassifier):
    name = "tfidf_rf"

    def _make_clf(self, hyperparams: dict):
        return RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 200),
            max_depth=hyperparams.get("max_depth", None),
            class_weight="balanced",
            n_jobs=-1,
        )


# ── Transformer placeholder ───────────────────────────────────────────────────

@register
class TransformerDetector(BaseDetector):
    """HuggingFace transformer placeholder. Implement when torch + transformers are installed."""

    name = "transformer"
    modality = "text"

    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics:
        raise NotImplementedError("Install torch + transformers and implement this method.")

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
