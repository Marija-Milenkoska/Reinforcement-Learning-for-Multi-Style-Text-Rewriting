from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path


HUMAN_EVALUATION_COLUMNS = (
    "timestamp_utc",
    "evaluator_id",
    "batch_id",
    "item_number",
    "original_text",
    "target_style",
    "generated_text",
    "meaning_score",
    "fluency_score",
    "style_match_score",
)
RATING_FIELDS = ("meaning_score", "fluency_score", "style_match_score")


def append_human_rating(path: str | Path, rating: dict[str, str | int | float]) -> dict[str, str | int | float]:
    """Validate and append one 1-5 human rating to the evaluation CSV."""
    row = dict(rating)
    for field in RATING_FIELDS:
        score = int(row[field])
        if not 1 <= score <= 5:
            raise ValueError(f"{field} must be between 1 and 5.")
        row[field] = score

    row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HUMAN_EVALUATION_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in HUMAN_EVALUATION_COLUMNS})
    return row


def load_human_ratings(path: str | Path) -> list[dict[str, str | int]]:
    ratings_path = Path(path)
    if not ratings_path.exists():
        return []

    rows: list[dict[str, str | int]] = []
    try:
        with open(ratings_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for field in RATING_FIELDS:
                    row[field] = int(row[field])
                row["item_number"] = int(row["item_number"])
                rows.append(row)
    except (KeyError, TypeError, ValueError, csv.Error):
        return []
    return rows


def summarize_human_ratings(ratings: list[dict[str, str | int | float]]) -> dict[str, float | int]:
    if not ratings:
        return {"count": 0, "meaning": 0.0, "fluency": 0.0, "style_match": 0.0}
    count = len(ratings)
    return {
        "count": count,
        "meaning": sum(float(row["meaning_score"]) for row in ratings) / count,
        "fluency": sum(float(row["fluency_score"]) for row in ratings) / count,
        "style_match": sum(float(row["style_match_score"]) for row in ratings) / count,
    }


def render_human_summary(summary: dict[str, float | int]) -> str:
    if not summary["count"]:
        return "No human ratings have been submitted yet."
    return (
        "### Human Evaluation Summary\n\n"
        "| Rated samples | Meaning (1-5) | Fluency (1-5) | Style match (1-5) |\n"
        "| ---: | ---: | ---: | ---: |\n"
        f"| {summary['count']} | {float(summary['meaning']):.2f} | "
        f"{float(summary['fluency']):.2f} | {float(summary['style_match']):.2f} |"
    )
