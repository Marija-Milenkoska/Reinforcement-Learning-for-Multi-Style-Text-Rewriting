import pytest

from rl_rewriter.human_evaluation import (
    append_human_rating,
    load_human_ratings,
    render_human_summary,
    summarize_human_ratings,
)


def _rating() -> dict[str, str | int]:
    return {
        "evaluator_id": "tester",
        "batch_id": "batch-1",
        "item_number": 1,
        "original_text": "The city added new bike lanes.",
        "target_style": "journalistic",
        "generated_text": "Officials announced new bike lanes.",
        "meaning_score": 4,
        "fluency_score": 5,
        "style_match_score": 3,
    }


def test_append_and_load_human_rating(tmp_path):
    output = tmp_path / "human_evaluations.csv"
    append_human_rating(output, _rating())

    ratings = load_human_ratings(output)

    assert len(ratings) == 1
    assert ratings[0]["meaning_score"] == 4
    assert ratings[0]["timestamp_utc"]


def test_human_rating_rejects_invalid_score(tmp_path):
    rating = _rating()
    rating["fluency_score"] = 6

    with pytest.raises(ValueError, match="between 1 and 5"):
        append_human_rating(tmp_path / "human_evaluations.csv", rating)


def test_human_summary_reports_average_scores():
    summary = summarize_human_ratings([_rating(), {**_rating(), "meaning_score": 2, "fluency_score": 3}])

    assert summary == {"count": 2, "meaning": 3.0, "fluency": 4.0, "style_match": 3.0}
    assert "3.00" in render_human_summary(summary)
