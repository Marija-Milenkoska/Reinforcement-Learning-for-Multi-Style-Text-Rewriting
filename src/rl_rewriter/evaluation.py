from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from .dataset import clean_dataset, load_dataset
from .pipeline import RewriterPipeline

logger = logging.getLogger(__name__)


def _bleu(reference: str, hypothesis: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        return sentence_bleu(
            [reference.lower().split()],
            hypothesis.lower().split(),
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return 0.0


def _rouge(reference: str, hypothesis: str) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer as rs

        scorer = rs.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def _meteor(reference: str, hypothesis: str) -> float:
    try:
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        from nltk.translate.meteor_score import meteor_score

        return float(meteor_score([reference.lower().split()], hypothesis.lower().split()))
    except Exception:
        return 0.0


@dataclass(slots=True)
class EvaluationSummary:
    model_name: str
    avg_style_score: float
    avg_meaning_score: float
    avg_fluency_score: float
    avg_reward: float
    avg_bleu: float
    avg_rouge1: float
    avg_rouge2: float
    avg_rougeL: float
    avg_meteor: float


def _nlg_scores(reference: str, hypothesis: str) -> dict[str, float]:
    rouge = _rouge(reference, hypothesis)
    return {
        "bleu": _bleu(reference, hypothesis),
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "meteor": _meteor(reference, hypothesis),
    }


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
        reference = record["styled_text"]
        nlg = _nlg_scores(reference, result.best_text)
        rows.append(
            {
                "id": record["id"],
                "original_text": record["original_text"],
                "target_style": record["target_style"],
                "reference_text": reference,
                "generated_text": result.best_text,
                "style_score": result.style_score,
                "meaning_score": result.meaning_score,
                "fluency_score": result.fluency_score,
                "total_reward": result.total_reward,
                **nlg,
            }
        )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    n = len(rows) or 1
    summary = EvaluationSummary(
        model_name="reward-selected-generator",
        avg_style_score=sum(float(r["style_score"]) for r in rows) / n,
        avg_meaning_score=sum(float(r["meaning_score"]) for r in rows) / n,
        avg_fluency_score=sum(float(r["fluency_score"]) for r in rows) / n,
        avg_reward=sum(float(r["total_reward"]) for r in rows) / n,
        avg_bleu=sum(float(r["bleu"]) for r in rows) / n,
        avg_rouge1=sum(float(r["rouge1"]) for r in rows) / n,
        avg_rouge2=sum(float(r["rouge2"]) for r in rows) / n,
        avg_rougeL=sum(float(r["rougeL"]) for r in rows) / n,
        avg_meteor=sum(float(r["meteor"]) for r in rows) / n,
    )
    logger.info("Evaluated %d records — avg reward: %.3f  BLEU: %.3f", len(rows), summary.avg_reward, summary.avg_bleu)
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
        reference = record["styled_text"]
        baseline = pipeline.rewrite_baseline(
            text=record["original_text"],
            target_style=record["target_style"],
        )
        reward_selected = pipeline.rewrite(
            text=record["original_text"],
            target_style=record["target_style"],
            num_candidates=num_candidates,
        )
        for strategy, result in (("baseline", baseline), ("reward-selected", reward_selected)):
            nlg = _nlg_scores(reference, result.best_text)
            rows.append(
                {
                    "id": record["id"],
                    "strategy": strategy,
                    "original_text": record["original_text"],
                    "target_style": record["target_style"],
                    "reference_text": reference,
                    "generated_text": result.best_text,
                    "style_score": result.style_score,
                    "meaning_score": result.meaning_score,
                    "fluency_score": result.fluency_score,
                    "total_reward": result.total_reward,
                    **nlg,
                }
            )

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    summaries: list[EvaluationSummary] = []
    for strategy in ("baseline", "reward-selected"):
        sr = [row for row in rows if row["strategy"] == strategy]
        n = len(sr) or 1
        summaries.append(
            EvaluationSummary(
                model_name=strategy,
                avg_style_score=sum(float(r["style_score"]) for r in sr) / n,
                avg_meaning_score=sum(float(r["meaning_score"]) for r in sr) / n,
                avg_fluency_score=sum(float(r["fluency_score"]) for r in sr) / n,
                avg_reward=sum(float(r["total_reward"]) for r in sr) / n,
                avg_bleu=sum(float(r["bleu"]) for r in sr) / n,
                avg_rouge1=sum(float(r["rouge1"]) for r in sr) / n,
                avg_rouge2=sum(float(r["rouge2"]) for r in sr) / n,
                avg_rougeL=sum(float(r["rougeL"]) for r in sr) / n,
                avg_meteor=sum(float(r["meteor"]) for r in sr) / n,
            )
        )
    return rows, summaries


def render_summary_markdown(summaries: list[EvaluationSummary]) -> str:
    lines = [
        "| Strategy | Style | Meaning | Fluency | Reward | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| {s.model_name} | {s.avg_style_score:.3f} | {s.avg_meaning_score:.3f} | "
            f"{s.avg_fluency_score:.3f} | {s.avg_reward:.3f} | {s.avg_bleu:.3f} | "
            f"{s.avg_rouge1:.3f} | {s.avg_rouge2:.3f} | {s.avg_rougeL:.3f} | {s.avg_meteor:.3f} |"
        )
    return "\n".join(lines)
