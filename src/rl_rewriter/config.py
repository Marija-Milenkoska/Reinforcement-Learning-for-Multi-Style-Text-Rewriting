from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CONFIG_FILE = Path(__file__).resolve().parents[2] / "config.yaml"


def _load_yaml() -> dict[str, Any]:
    if not _CONFIG_FILE.exists():
        return {}
    try:
        import yaml
        with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


@dataclass(slots=True)
class ModelConfig:
    generator_model_name: str = "google/flan-t5-small"
    fine_tuned_generator_path: Path = Path("models/flan-t5-finetuned")
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    style_classifier_name: str = "distilbert-base-uncased"
    style_classifier_path: Path = Path("models/style-classifier/style_classifier.pkl")
    device: str = "cpu"
    max_new_tokens: int = 72


@dataclass(slots=True)
class RewardWeights:
    style: float = 0.45
    meaning: float = 0.40
    fluency: float = 0.15


@dataclass(slots=True)
class ProjectConfig:
    styles: tuple[str, ...] = ("poetic", "journalistic", "formal")
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        cfg = _load_yaml()
        if not cfg:
            return

        if "reward_weights" in cfg:
            rw = cfg["reward_weights"]
            self.reward_weights = RewardWeights(
                style=float(rw.get("style", self.reward_weights.style)),
                meaning=float(rw.get("meaning", self.reward_weights.meaning)),
                fluency=float(rw.get("fluency", self.reward_weights.fluency)),
            )

        if "model" in cfg:
            m = cfg["model"]
            self.model = ModelConfig(
                generator_model_name=m.get("generator_model_name", self.model.generator_model_name),
                sentence_model_name=m.get("sentence_model_name", self.model.sentence_model_name),
                device=m.get("device", self.model.device),
                max_new_tokens=int(m.get("max_new_tokens", self.model.max_new_tokens)),
                fine_tuned_generator_path=self.model.fine_tuned_generator_path,
                style_classifier_name=self.model.style_classifier_name,
                style_classifier_path=self.model.style_classifier_path,
            )

        if "styles" in cfg:
            self.styles = tuple(cfg["styles"])
