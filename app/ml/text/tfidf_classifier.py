"""TF-IDF based scam detectors — fast, interpretable baselines.

Two variants registered:
  - tfidf_logreg  (Logistic Regression — best calibrated probabilities)
  - tfidf_svm     (LinearSVC — often best raw accuracy on text)
  - tfidf_rf      (Random Forest — good for noisy/imbalanced data)

Add more by subclassing TfidfClassifier and changing `name` and `_make_clf`.
"""
from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from app.ml.base import BaseDetector, TrainingMetrics
from app.ml.registry import register


class TfidfClassifier(BaseDetector):
    """Base for TF-IDF pipeline classifiers. Subclass and set `name` + `_make_clf()`."""

    modality = "text"
    _pipeline: Pipeline | None = None

    def _make_clf(self, hyperparams: dict):
        raise NotImplementedError

    def train(self, X: list[str], y: list[int], hyperparams: dict) -> TrainingMetrics:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        max_features = hyperparams.get("max_features", 20_000)
        ngram_range = tuple(hyperparams.get("ngram_range", [1, 2]))

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
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
        confidences = self._binary_confidence(contents)
        return [(bool(c >= 0.5), c) for c in confidences]

    def explain(self, content: str) -> list[str]:
        """Return top TF-IDF feature tokens for this input."""
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

    # -- helpers --

    def _get_proba(self, X: list[str]) -> np.ndarray | None:
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self._pipeline.predict_proba(X)[:, 1]
        return None

    def _binary_confidence(self, X: list[str]) -> list[float]:
        """Fallback for classifiers without predict_proba (e.g. raw LinearSVC)."""
        preds = self._pipeline.predict(X)
        return [float(p) for p in preds]

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
        # Wrap in CalibratedClassifierCV to get probability estimates
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
