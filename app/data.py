"""Data utilities: text preprocessing and dataset-agnostic loading."""
import html
import re

import pandas as pd
from pathlib import Path

from app.config import ColumnMap


# ── Preprocessing ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize a raw text message or email body for ML consumption."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\+?[\d][\d\s\-().]{7,}\d", " PHONE ", text)
    text = re.sub(r"\$[\d,]+(\.\d+)?", " MONEY ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_email_parts(raw_email: str) -> dict:
    """Split a raw email string into header fields and body.

    Returns a dict with keys: subject, sender, body, cleaned_body.
    If the input is a plain text message (no headers), body == raw_email.
    """
    subject = ""
    sender = ""
    body = raw_email

    lines = raw_email.split("\n")
    in_headers = True
    for i, line in enumerate(lines):
        if in_headers:
            if line.strip() == "":
                body = "\n".join(lines[i + 1:])
                in_headers = False
            elif line.lower().startswith("subject:"):
                subject = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("from:"):
                sender = line.split(":", 1)[-1].strip()

    return {
        "subject": subject,
        "sender": sender,
        "body": body,
        "cleaned_body": clean_text(subject + " " + body),
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

class DatasetLoadError(Exception):
    pass


def load_dataset(
    path: Path,
    fmt: str,
    column_map: ColumnMap,
    preprocess: bool = True,
) -> tuple[list[str], list[int]]:
    """Load a CSV or JSON dataset and return (texts, binary_labels).

    Args:
        path: Path to the file.
        fmt: "csv" or "json".
        column_map: Maps logical names to actual column names.
        preprocess: If True, run clean_text() on each sample.

    Returns:
        texts: list of strings (one per sample).
        labels: list of 0/1 integers (1 = scam/spam).

    Raises:
        DatasetLoadError: If the file cannot be read or required columns are missing.
    """
    try:
        if fmt == "csv":
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin-1')
        elif fmt == "json":
            df = pd.read_json(path)
        else:
            raise DatasetLoadError(f"Unsupported format '{fmt}'. Use 'csv' or 'json'.")
    except DatasetLoadError:
        raise
    except Exception as e:
        raise DatasetLoadError(f"Failed to read dataset at {path}: {e}") from e

    _check_columns(df, column_map)

    texts = df[column_map.text_column].astype(str).tolist()
    raw_labels = df[column_map.label_column].astype(str).tolist()
    labels = [
        1 if lbl.strip().lower() == column_map.label_positive_value.strip().lower() else 0
        for lbl in raw_labels
    ]

    if preprocess:
        texts = [clean_text(t) for t in texts]

    _validate_labels(labels)
    return texts, labels


def _check_columns(df: pd.DataFrame, column_map: ColumnMap) -> None:
    available = list(df.columns)
    for logical, actual in [("text", column_map.text_column), ("label", column_map.label_column)]:
        if actual not in df.columns:
            raise DatasetLoadError(
                f"Expected {logical} column '{actual}' not found in dataset. "
                f"Available columns: {available}"
            )


def _validate_labels(labels: list[int]) -> None:
    pos = sum(labels)
    total = len(labels)
    if total == 0:
        raise DatasetLoadError("Dataset is empty after loading.")
    if pos == 0:
        raise DatasetLoadError(
            "No positive (scam) examples found. Check label_positive_value in your column_map."
        )
    if pos == total:
        raise DatasetLoadError("All examples are positive — dataset may be mislabeled.")
