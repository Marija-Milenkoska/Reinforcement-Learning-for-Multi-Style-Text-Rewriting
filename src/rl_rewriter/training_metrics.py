from __future__ import annotations

import csv
from pathlib import Path


METRIC_COLUMNS = (
    "epoch",
    "avg_reward",
    "avg_loss",
    "avg_style_score",
    "avg_meaning_score",
    "avg_fluency_score",
    "baseline",
    "examples",
)


def write_training_metrics(path: str | Path, metrics: list[dict[str, float | int]]) -> None:
    """Replace the dashboard CSV with metrics recorded for the current training run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in metrics:
            writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def load_training_metrics(path: str | Path) -> list[dict[str, float | int]]:
    """Load valid epoch records without making the dashboard fail on a partial CSV."""
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []

    rows: list[dict[str, float | int]] = []
    try:
        with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "avg_reward": float(row["avg_reward"]),
                        "avg_loss": float(row["avg_loss"]),
                        "avg_style_score": float(row["avg_style_score"]),
                        "avg_meaning_score": float(row["avg_meaning_score"]),
                        "avg_fluency_score": float(row["avg_fluency_score"]),
                        "baseline": float(row["baseline"]),
                        "examples": int(row["examples"]),
                    }
                )
    except (KeyError, TypeError, ValueError, csv.Error):
        return []
    return rows
