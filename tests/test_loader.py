"""Tests for the dataset-agnostic loader."""
import pytest
import pandas as pd
from pathlib import Path

from app.config import ColumnMap
from app.data import load_dataset, DatasetLoadError


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "message": ["Win a prize now!", "Hey, call me later", "Click here to claim $1000"],
        "category": ["spam", "ham", "spam"],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def make_col_map(text="message", label="category", pos="spam"):
    return ColumnMap(text_column=text, label_column=label, label_positive_value=pos)


def test_loads_csv_correctly(sample_csv):
    X, y = load_dataset(sample_csv, "csv", make_col_map())
    assert len(X) == 3
    assert y == [1, 0, 1]


def test_preprocessing_runs(sample_csv):
    X, _ = load_dataset(sample_csv, "csv", make_col_map(), preprocess=True)
    assert all(s == s.lower() for s in X)


def test_wrong_text_column_raises(sample_csv):
    with pytest.raises(DatasetLoadError, match="text column 'body'"):
        load_dataset(sample_csv, "csv", make_col_map(text="body"))


def test_wrong_label_column_raises(sample_csv):
    with pytest.raises(DatasetLoadError, match="label column 'label'"):
        load_dataset(sample_csv, "csv", make_col_map(label="label"))


def test_wrong_positive_value_raises(sample_csv):
    with pytest.raises(DatasetLoadError, match="No positive"):
        load_dataset(sample_csv, "csv", make_col_map(pos="SCAM"))
