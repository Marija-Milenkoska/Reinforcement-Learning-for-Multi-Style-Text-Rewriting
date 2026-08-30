from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .dataset import clean_dataset, load_dataset
from .pipeline import RewriterPipeline


@dataclass(slots=True)
class EvaluationSummary:
    model_name: str
    avg_style_score: float
    avg_meaning_score: float
    avg_fluency_score: float
    avg_reward: float


def evaluate_dataset(
    pipeline: RewriterPipeline,
    dataset_path: str | Path,
    output_path: str | Path,
    num_candidates: int = 5,
) -> tuple[list[dict[str, str | float]], EvaluationSummary]:
    dataset = clean_dataset(load_dataset(dataset_path))
    rows = []

    for record in dataset:
        result = pipeline.rewrite(
            text=record["original_text"],
            target_style=record["target_style"],
            num_candidates=num_candidates,
        )
        rows.append(
            {
                "id": record["id"],
                "original_text": record["original_text"],
                "target_style": record["target_style"],
                "reference_text": record["styled_text"],
                "generated_text": result.best_text,
                "style_score": result.style_score,
                "meaning_score": result.meaning_score,
                "fluency_score": result.fluency_score,
                "total_reward": result.total_reward,
            }
        )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    total_rows = len(rows) or 1
    summary = EvaluationSummary(
        model_name="reward-selected-generator",
        avg_style_score=sum(float(row["style_score"]) for row in rows) / total_rows,
        avg_meaning_score=sum(float(row["meaning_score"]) for row in rows) / total_rows,
        avg_fluency_score=sum(float(row["fluency_score"]) for row in rows) / total_rows,
        avg_reward=sum(float(row["total_reward"]) for row in rows) / total_rows,
    )
    return rows, summary


def compare_generation_strategies(
    pipeline: RewriterPipeline,
    dataset_path: str | Path,
    output_path: str | Path,
    num_candidates: int = 5,
) -> tuple[list[dict[str, str | float]], list[EvaluationSummary]]:
    dataset = clean_dataset(load_dataset(dataset_path))
    rows: list[dict[str, str | float]] = []

    for record in dataset:
        baseline = pipeline.rewrite_baseline(
            text=record["original_text"],
            target_style=record["target_style"],
        )
        reward_selected = pipeline.rewrite(
            text=record["original_text"],
            target_style=record["target_style"],
            num_candidates=num_candidates,
        )
        rows.extend(
            [
                {
                    "id": record["id"],
                    "strategy": "baseline",
                    "original_text": record["original_text"],
                    "target_style": record["target_style"],
                    "reference_text": record["styled_text"],
                    "generated_text": baseline.best_text,
                    "style_score": baseline.style_score,
                    "meaning_score": baseline.meaning_score,
                    "fluency_score": baseline.fluency_score,
                    "total_reward": baseline.total_reward,
                },
                {
                    "id": record["id"],
                    "strategy": "reward-selected",
                    "original_text": record["original_text"],
                    "target_style": record["target_style"],
                    "reference_text": record["styled_text"],
                    "generated_text": reward_selected.best_text,
                    "style_score": reward_selected.style_score,
                    "meaning_score": reward_selected.meaning_score,
                    "fluency_score": reward_selected.fluency_score,
                    "total_reward": reward_selected.total_reward,
                },
            ]
        )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summaries: list[EvaluationSummary] = []
    for strategy in ("baseline", "reward-selected"):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        total_rows = len(strategy_rows) or 1
        summaries.append(
            EvaluationSummary(
                model_name=strategy,
                avg_style_score=sum(float(row["style_score"]) for row in strategy_rows) / total_rows,
                avg_meaning_score=sum(float(row["meaning_score"]) for row in strategy_rows) / total_rows,
                avg_fluency_score=sum(float(row["fluency_score"]) for row in strategy_rows) / total_rows,
                avg_reward=sum(float(row["total_reward"]) for row in strategy_rows) / total_rows,
            )
        )
    return rows, summaries


def render_summary_markdown(summaries: list[EvaluationSummary]) -> str:
    lines = [
        "| Strategy | Style | Meaning | Fluency | Reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.model_name} | {summary.avg_style_score:.3f} | "
            f"{summary.avg_meaning_score:.3f} | {summary.avg_fluency_score:.3f} | "
            f"{summary.avg_reward:.3f} |"
        )
    return "\n".join(lines)
