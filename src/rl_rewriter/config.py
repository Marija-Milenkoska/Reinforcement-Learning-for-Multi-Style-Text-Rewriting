from dataclasses import dataclass, field
from pathlib import Path


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
