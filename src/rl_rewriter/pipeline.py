from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig
from .generator import Candidate, TextGenerator
from .scoring import RewardScorer, ScoreBreakdown


@dataclass(slots=True)
class RankedCandidate:
    text: str
    style_score: float
    meaning_score: float
    fluency_score: float
    total_reward: float


@dataclass(slots=True)
class RewriteResult:
    original_text: str
    target_style: str
    best_text: str
    style_score: float
    meaning_score: float
    fluency_score: float
    total_reward: float
    style_backend: str
    meaning_backend: str
    candidates: list[RankedCandidate]


class RewriterPipeline:
    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        self.generator = TextGenerator(self.config.model)
        self.reward_scorer = RewardScorer(self.config.reward_weights, self.config.model)

    def rewrite(self, text: str, target_style: str, num_candidates: int = 5) -> RewriteResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text must not be empty.")
        if target_style not in self.config.styles:
            raise ValueError(f"Unsupported style: {target_style}")

        generated_candidates = self.generator.generate_candidates(
            text=normalized_text,
            target_style=target_style,
            num_candidates=num_candidates,
        )
        ranked_candidates, best_score = self._rank_candidates(normalized_text, target_style, generated_candidates)
        best_candidate = ranked_candidates[0]
        return RewriteResult(
            original_text=normalized_text,
            target_style=target_style,
            best_text=best_candidate.text,
            style_score=best_candidate.style_score,
            meaning_score=best_candidate.meaning_score,
            fluency_score=best_candidate.fluency_score,
            total_reward=best_candidate.total_reward,
            style_backend=best_score.style_backend,
            meaning_backend=best_score.meaning_backend,
            candidates=ranked_candidates,
        )

    def rewrite_baseline(self, text: str, target_style: str) -> RewriteResult:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Input text must not be empty.")
        generated_candidates = self.generator.generate_candidates(
            text=normalized_text,
            target_style=target_style,
            num_candidates=1,
        )
        candidate = generated_candidates[0]
        score = self.reward_scorer.calculate_reward(normalized_text, candidate.text, target_style)
        ranked_candidate = self._to_ranked_candidate(candidate.text, score)
        return RewriteResult(
            original_text=normalized_text,
            target_style=target_style,
            best_text=ranked_candidate.text,
            style_score=ranked_candidate.style_score,
            meaning_score=ranked_candidate.meaning_score,
            fluency_score=ranked_candidate.fluency_score,
            total_reward=ranked_candidate.total_reward,
            style_backend=score.style_backend,
            meaning_backend=score.meaning_backend,
            candidates=[ranked_candidate],
        )

    def compare_all_styles(self, text: str, num_candidates: int = 5) -> list[RewriteResult]:
        return [self.rewrite(text=text, target_style=style, num_candidates=num_candidates) for style in self.config.styles]

    def _rank_candidates(
        self,
        original_text: str,
        target_style: str,
        candidates: list[Candidate],
    ) -> tuple[list[RankedCandidate], ScoreBreakdown]:
        ranked: list[tuple[RankedCandidate, ScoreBreakdown]] = []
        for candidate in candidates:
            score = self.reward_scorer.calculate_reward(original_text, candidate.text, target_style)
            ranked.append((self._to_ranked_candidate(candidate.text, score), score))
        sorted_ranked = sorted(ranked, key=lambda item: item[0].total_reward, reverse=True)
        return [item[0] for item in sorted_ranked], sorted_ranked[0][1]

    @staticmethod
    def _to_ranked_candidate(text: str, score: ScoreBreakdown) -> RankedCandidate:
        return RankedCandidate(
            text=text,
            style_score=score.style,
            meaning_score=score.meaning,
            fluency_score=score.fluency,
            total_reward=score.total,
        )

    @staticmethod
    def format_candidates(result: RewriteResult) -> str:
        lines = []
        for index, candidate in enumerate(result.candidates, start=1):
            lines.append(
                f"{index}. reward={candidate.total_reward:.3f} | "
                f"style={candidate.style_score:.3f} | "
                f"meaning={candidate.meaning_score:.3f} | "
                f"fluency={candidate.fluency_score:.3f}\n{candidate.text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def load_examples(path: str | Path) -> list[dict[str, str]]:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
