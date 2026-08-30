from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import exp
import re

from .config import ModelConfig, RewardWeights
from .style_classifier import StyleClassifier


STYLE_KEYWORDS = {
    "poetic": {"sky", "whispered", "gentle", "verse", "rhythm", "light"},
    "journalistic": {"report", "officials", "authorities", "announcement", "information"},
    "formal": {"evidence", "indicates", "academic", "perspective", "conclude", "therefore"},
}


def _normalize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _jaccard_similarity(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


@dataclass(slots=True)
class ScoreBreakdown:
    style: float
    meaning: float
    fluency: float
    total: float
    style_backend: str
    meaning_backend: str


class RewardScorer:
    def __init__(self, weights: RewardWeights, model_config: ModelConfig):
        self.weights = weights
        self.model_config = model_config
        self._semantic_model = None
        self._meaning_backend = "token-jaccard"
        self.style_classifier = StyleClassifier(model_config.style_classifier_path)

        try:
            from sentence_transformers import SentenceTransformer

            self._semantic_model = SentenceTransformer(model_config.sentence_model_name)
            self._meaning_backend = model_config.sentence_model_name
        except Exception:  # pragma: no cover - fallback path is intentional
            self._semantic_model = None

    def style_score(self, text: str, target_style: str) -> float:
        model_score = self.style_classifier.score(text, target_style)
        tokens = set(_normalize(text))
        keywords = STYLE_KEYWORDS[target_style]
        hit_rate = len(tokens & keywords) / max(len(keywords), 1)
        return min(1.0, 0.75 * model_score + 0.25 * hit_rate)

    def meaning_score(self, original: str, generated: str) -> float:
        if self._semantic_model is not None:
            from sentence_transformers.util import cos_sim

            embeddings = self._semantic_model.encode([original, generated], convert_to_tensor=True)
            similarity = float(cos_sim(embeddings[0], embeddings[1]).item())
            return max(0.0, min((similarity + 1) / 2, 1.0))

        return _jaccard_similarity(_normalize(original), _normalize(generated))

    def fluency_score(self, text: str) -> float:
        tokens = _normalize(text)
        if not tokens:
            return 0.0

        token_count = len(tokens)
        counts = Counter(tokens)
        repetition_ratio = max(counts.values()) / token_count
        repetition_penalty = max(0.0, 1 - repetition_ratio)
        length_score = exp(-abs(token_count - 18) / 18)
        punctuation_bonus = 0.1 if text.strip().endswith((".", "!", "?")) else 0.0
        return min(1.0, 0.55 * repetition_penalty + 0.35 * length_score + punctuation_bonus)

    def calculate_reward(self, original: str, generated: str, target_style: str) -> ScoreBreakdown:
        style = self.style_score(generated, target_style)
        meaning = self.meaning_score(original, generated)
        fluency = self.fluency_score(generated)
        total = (
            self.weights.style * style
            + self.weights.meaning * meaning
            + self.weights.fluency * fluency
        )
        return ScoreBreakdown(
            style=style,
            meaning=meaning,
            fluency=fluency,
            total=total,
            style_backend=self.style_classifier.backend,
            meaning_backend=self._meaning_backend,
        )
