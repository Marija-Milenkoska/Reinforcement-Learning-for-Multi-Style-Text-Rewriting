from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from .config import ModelConfig


PROMPTS = {
    "poetic": "Rewrite the following text in a poetic style:\n{text}",
    "journalistic": "Rewrite the following text in a journalistic style:\n{text}",
    "formal": "Rewrite the following text in a formal academic style:\n{text}",
}


@dataclass(slots=True)
class Candidate:
    text: str
    metadata: dict[str, Any]


class TextGenerator:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._pipeline = None
        self._load_error = None
        self._model_source = "fallback"

        try:
            from transformers import pipeline

            model_name = self._resolve_generator_model()
            self._pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=config.device,
            )
            self._model_source = str(model_name)
        except Exception as exc:  # pragma: no cover - fallback path is intentional
            self._load_error = exc

    def _resolve_generator_model(self) -> str:
        fine_tuned_path = Path(self.config.fine_tuned_generator_path)
        if fine_tuned_path.exists():
            return str(fine_tuned_path)
        return self.config.generator_model_name

    def generate_candidates(self, text: str, target_style: str, num_candidates: int) -> list[Candidate]:
        if self._pipeline is None:
            return self._fallback_generate(text=text, target_style=target_style, num_candidates=num_candidates)

        prompt = PROMPTS[target_style].format(text=text.strip())
        outputs = self._pipeline(
            [prompt] * num_candidates,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=1,
            max_new_tokens=self.config.max_new_tokens,
        )
        return [
            Candidate(
                text=output["generated_text"].strip(),
                metadata={"source": "transformers", "model": self._model_source},
            )
            for output in outputs
        ]

    def _fallback_generate(self, text: str, target_style: str, num_candidates: int) -> list[Candidate]:
        base_text = text.strip().rstrip(".")
        normalized_text = " ".join(base_text.split())
        lowered_text = normalized_text.lower()
        variants = {
            "poetic": [
                f"Beneath a softer sky, {lowered_text}.",
                f"In the hush of the moment, {lowered_text}.",
                f"The air carries this change gently: {lowered_text}.",
                f"With a calm and steady rhythm, {lowered_text}.",
                f"As though evening light had touched it, {lowered_text}.",
                f"A quieter world begins where {lowered_text}.",
                f"Across the horizon of the day, {lowered_text}.",
                f"The moment opens like a small song: {lowered_text}.",
            ],
            "journalistic": [
                f"Officials stated that {lowered_text}.",
                f"The latest report confirms that {lowered_text}.",
                f"According to the announcement, {lowered_text}.",
                f"Public officials announced that {lowered_text}.",
                f"New information shows that {lowered_text}.",
                f"The statement said that {lowered_text}.",
                f"Early reporting indicates that {lowered_text}.",
                f"The update noted that {lowered_text}.",
            ],
            "formal": [
                f"It can be observed that {lowered_text}.",
                f"The available evidence indicates that {lowered_text}.",
                f"This development suggests that {lowered_text}.",
                f"From an academic perspective, {lowered_text}.",
                f"It is therefore reasonable to conclude that {lowered_text}.",
                f"A formal analysis suggests that {lowered_text}.",
                f"This outcome may be interpreted as evidence that {lowered_text}.",
                f"In practical terms, the findings indicate that {lowered_text}.",
                f"This observation supports the conclusion that {lowered_text}.",
                f"It is appropriate to note that {lowered_text}.",
            ],
        }

        style_variants = variants[target_style]
        offset = int(hashlib.md5(normalized_text.encode("utf-8")).hexdigest(), 16) % len(style_variants)
        candidates = []
        for index in range(num_candidates):
            rewritten = style_variants[(index + offset) % len(style_variants)]
            candidates.append(Candidate(text=rewritten, metadata={"source": "fallback", "rank": index + 1}))
        return candidates
