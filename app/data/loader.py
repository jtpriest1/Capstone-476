"""Dataset-agnostic loader. Reads CSV or JSON into (texts, labels) using a column map."""
import pandas as pd
from pathlib import Path
from app.config import ColumnMap
from app.data.preprocessor import clean_text


class DatasetLoadError(Exception):
    pass


def load_dataset(
    path: Path,
    fmt: str,  # "csv" or "json"
    column_map: ColumnMap,
    preprocess: bool = True,
) -> tuple[list[str], list[int]]:
    """Load a dataset file and return (texts, binary_labels).

    Args:
        path: Absolute path to the file.
        fmt: "csv" or "json".
        column_map: Maps logical names to actual column names in the file.
        preprocess: If True, run clean_text() on each sample.

    Returns:
        texts: list of strings (one per sample).
        labels: list of 0/1 integers (1 = scam/spam).

    Raises:
        DatasetLoadError: If the file cannot be read or required columns are missing.
    """
    try:
        if fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "json":
            df = pd.read_json(path)
        else:
            raise DatasetLoadError(f"Unsupported format '{fmt}'. Use 'csv' or 'json'.")
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
    for logical, actual in [
        ("text", column_map.text_column),
        ("label", column_map.label_column),
    ]:
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
