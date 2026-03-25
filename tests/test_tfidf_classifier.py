"""Tests for the TF-IDF classifier — no disk I/O required."""
import pytest
from app.model import TfidfLogReg


SPAM = [
    "Congratulations you won $1000 click here to claim your prize",
    "Free money transfer act now limited time offer",
    "You have been selected for a cash reward click this link",
    "Win big prizes enter now free lottery winner selected",
    "Urgent your account will be closed verify now",
]
HAM = [
    "Hey can we meet for lunch tomorrow",
    "The project deadline is next Friday please review the doc",
    "I will call you back later this afternoon",
    "Reminder your dentist appointment is on Monday",
    "Thanks for the birthday wishes see you at the party",
]


@pytest.fixture
def trained_model():
    model = TfidfLogReg()
    model.train(SPAM + HAM, [1] * len(SPAM) + [0] * len(HAM), {})
    return model


def test_predict_spam(trained_model):
    is_scam, confidence = trained_model.predict("You won a free prize click here now")
    assert is_scam is True
    assert 0.0 <= confidence <= 1.0


def test_predict_ham(trained_model):
    is_scam, _ = trained_model.predict("See you at the meeting tomorrow")
    assert is_scam is False


def test_predict_batch(trained_model):
    results = trained_model.predict_batch(["Win free money", "Call me later"])
    assert len(results) == 2
    assert results[0][0] is True
    assert results[1][0] is False


def test_explain_returns_tokens(trained_model):
    tokens = trained_model.explain("You won a free prize")
    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_untrained_raises(tmp_path):
    model = TfidfLogReg()
    with pytest.raises(RuntimeError, match="not trained"):
        model.predict("test")


def test_save_load_roundtrip(trained_model, tmp_path):
    path = tmp_path / "tfidf_logreg.joblib"
    trained_model.save(path)
    loaded = TfidfLogReg.load(path)
    is_scam, _ = loaded.predict("Win free money now click here")
    assert is_scam is True
