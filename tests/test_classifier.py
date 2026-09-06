from pathlib import Path

import pytest

from rl_rewriter.style_classifier import (
    NaiveBayesStyleModel,
    StyleClassifier,
    _tokenize,
    train_and_save_style_classifier,
)

CLASSIFIER_PATH = Path("models/style-classifier/style_classifier.pkl")
TRAIN_PATH = Path("data/processed/train.csv")


# --- _tokenize ---

def test_tokenize_lowercases():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_preserves_apostrophes():
    assert "it's" in _tokenize("it's a test")


def test_tokenize_empty():
    assert _tokenize("") == []


# --- NaiveBayesStyleModel ---

def _make_tiny_model() -> NaiveBayesStyleModel:
    class_priors = {"poetic": 0.5, "formal": 0.5}
    token_probs = {
        "poetic": {"sky": 0.8, "light": 0.6, "__unknown__": 0.01},
        "formal": {"evidence": 0.8, "therefore": 0.6, "__unknown__": 0.01},
    }
    return NaiveBayesStyleModel(
        class_priors=class_priors,
        token_probs=token_probs,
        vocabulary_size=10,
    )


def test_predict_proba_sums_to_one():
    model = _make_tiny_model()
    probs = model.predict_proba(["the sky was light"])[0]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_predict_proba_returns_one_per_class():
    model = _make_tiny_model()
    probs = model.predict_proba(["some text"])[0]
    assert len(probs) == 2


def test_predict_proba_favors_matching_class():
    model = _make_tiny_model()
    # classes_ = ["poetic", "formal"] — poetic first
    probs = model.predict_proba(["sky light sky"])[0]
    assert probs[0] > probs[1]


def test_predict_proba_unknown_tokens_handled():
    model = _make_tiny_model()
    probs = model.predict_proba(["completelymadeupword"])[0]
    assert abs(sum(probs) - 1.0) < 1e-6


# --- StyleClassifier ---

def test_classifier_loads_saved_model():
    if not CLASSIFIER_PATH.exists():
        pytest.skip("Trained model not present")
    clf = StyleClassifier(model_path=CLASSIFIER_PATH)
    assert clf.backend == "naive-bayes-token"
    score = clf.score("Officials confirmed the announcement.", "journalistic")
    assert 0.0 <= score <= 1.0


def test_classifier_heuristic_fallback_backend():
    clf = StyleClassifier(model_path=Path("nonexistent/model.pkl"))
    assert clf.backend == "heuristic-keywords"


def test_classifier_heuristic_no_keywords_is_baseline():
    clf = StyleClassifier(model_path=Path("nonexistent/model.pkl"))
    score = clf.score("A completely neutral sentence.", "poetic")
    assert score == pytest.approx(0.2)


def test_classifier_heuristic_keywords_raise_score():
    clf = StyleClassifier(model_path=Path("nonexistent/model.pkl"))
    score_with = clf.score("The sky was light and gentle.", "poetic")
    score_without = clf.score("A neutral sentence.", "poetic")
    assert score_with > score_without


def test_classifier_score_range_all_styles():
    clf = StyleClassifier(model_path=Path("nonexistent/model.pkl"))
    for style in ("poetic", "journalistic", "formal"):
        score = clf.score("Some sample text for testing.", style)
        assert 0.0 <= score <= 1.0


# --- train_and_save_style_classifier ---

def test_train_saves_file(tmp_path):
    if not TRAIN_PATH.exists():
        pytest.skip("Training data not present")
    output = tmp_path / "clf.pkl"
    info = train_and_save_style_classifier(dataset_paths=[TRAIN_PATH], output_path=output)
    assert output.exists()
    assert info["backend"] == "naive-bayes-token"
    assert info["num_examples"] > 0


def test_trained_model_is_loadable(tmp_path):
    if not TRAIN_PATH.exists():
        pytest.skip("Training data not present")
    output = tmp_path / "clf.pkl"
    train_and_save_style_classifier(dataset_paths=[TRAIN_PATH], output_path=output)
    clf = StyleClassifier(model_path=output)
    assert clf.backend == "naive-bayes-token"
    assert 0.0 <= clf.score("Officials reported the news.", "journalistic") <= 1.0
