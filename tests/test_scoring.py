from pathlib import Path

import pytest

from rl_rewriter.config import ModelConfig, RewardWeights
from rl_rewriter.scoring import RewardScorer, _jaccard_similarity, _normalize

CLASSIFIER_PATH = Path("models/style-classifier/style_classifier.pkl")


def make_scorer(weights: RewardWeights | None = None) -> RewardScorer:
    config = ModelConfig(style_classifier_path=CLASSIFIER_PATH)
    return RewardScorer(weights=weights or RewardWeights(), model_config=config)


# --- _normalize ---

def test_normalize_lowercases():
    assert _normalize("Hello World") == ["hello", "world"]


def test_normalize_strips_punctuation():
    tokens = _normalize("it's a test.")
    assert "it's" in tokens
    assert "test" in tokens
    assert "." not in tokens


def test_normalize_empty():
    assert _normalize("") == []


# --- _jaccard_similarity ---

def test_jaccard_identical():
    assert _jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0


def test_jaccard_disjoint():
    assert _jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0


def test_jaccard_partial():
    score = _jaccard_similarity(["a", "b", "c"], ["b", "c", "d"])
    assert 0.0 < score < 1.0


def test_jaccard_empty_left():
    assert _jaccard_similarity([], ["a"]) == 0.0


def test_jaccard_empty_right():
    assert _jaccard_similarity(["a"], []) == 0.0


# --- fluency_score ---

def test_fluency_empty_string():
    assert make_scorer().fluency_score("") == 0.0


def test_fluency_with_end_punctuation():
    score = make_scorer().fluency_score("The city introduced new bike lanes.")
    assert 0.0 <= score <= 1.0


def test_fluency_repetitive_lower_than_varied():
    scorer = make_scorer()
    repetitive = "the the the the the the the the the"
    varied = "The city introduced new bike lanes today."
    assert scorer.fluency_score(repetitive) < scorer.fluency_score(varied)


def test_fluency_range():
    scorer = make_scorer()
    for text in ["Hello.", "A sentence with some words.", "x " * 50]:
        assert 0.0 <= scorer.fluency_score(text) <= 1.0


# --- style_score ---

def test_style_score_range():
    scorer = make_scorer()
    for style in ("poetic", "journalistic", "formal"):
        score = scorer.style_score("Officials confirmed the new report.", style)
        assert 0.0 <= score <= 1.0


def test_style_score_keyword_boost_journalistic():
    scorer = make_scorer()
    score_with = scorer.style_score("Officials report the announcement.", "journalistic")
    score_without = scorer.style_score("The sky is blue and gentle.", "journalistic")
    assert score_with >= score_without


# --- meaning_score ---

def test_meaning_identical_text():
    scorer = make_scorer()
    text = "The city introduced new bike lanes."
    assert scorer.meaning_score(text, text) > 0.8


def test_meaning_range():
    scorer = make_scorer()
    score = scorer.meaning_score("The cat sat on a mat.", "The dog ran in the park.")
    assert 0.0 <= score <= 1.0


# --- calculate_reward ---

def test_calculate_reward_fields():
    scorer = make_scorer()
    breakdown = scorer.calculate_reward(
        "The city introduced new bike lanes.",
        "City officials confirmed new cycling infrastructure.",
        "journalistic",
    )
    assert 0.0 <= breakdown.style <= 1.0
    assert 0.0 <= breakdown.meaning <= 1.0
    assert 0.0 <= breakdown.fluency <= 1.0
    assert 0.0 <= breakdown.total <= 1.0
    assert breakdown.style_backend != ""
    assert breakdown.meaning_backend != ""


def test_calculate_reward_weights_applied():
    weights = RewardWeights(style=1.0, meaning=0.0, fluency=0.0)
    scorer = make_scorer(weights)
    breakdown = scorer.calculate_reward(
        "The city introduced new bike lanes.",
        "Officials reported new bike infrastructure.",
        "journalistic",
    )
    assert abs(breakdown.total - breakdown.style) < 1e-9
