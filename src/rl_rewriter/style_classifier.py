from __future__ import annotations

from collections import Counter, defaultdict
import math
import pickle
from pathlib import Path
import re

from .dataset import clean_dataset, load_dataset


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


class NaiveBayesStyleModel:
    def __init__(self, class_priors: dict[str, float], token_probs: dict[str, dict[str, float]], vocabulary_size: int):
        self.class_priors = class_priors
        self.token_probs = token_probs
        self.vocabulary_size = vocabulary_size
        self.classes_ = list(class_priors.keys())

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        outputs: list[list[float]] = []
        for text in texts:
            tokens = _tokenize(text)
            log_scores: dict[str, float] = {}
            for label in self.classes_:
                score = math.log(self.class_priors[label])
                label_probs = self.token_probs[label]
                unknown_prob = label_probs["__unknown__"]
                for token in tokens:
                    score += math.log(label_probs.get(token, unknown_prob))
                log_scores[label] = score

            max_log_score = max(log_scores.values())
            exp_scores = {label: math.exp(score - max_log_score) for label, score in log_scores.items()}
            total = sum(exp_scores.values()) or 1.0
            outputs.append([exp_scores[label] / total for label in self.classes_])
        return outputs


class StyleClassifier:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self.backend = "heuristic-keywords"
        self.labels = ["neutral", "poetic", "journalistic", "formal"]
        self._model = None

        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as handle:
                    payload = pickle.load(handle)
                self.labels = payload["labels"]
                self.backend = payload.get("backend", "naive-bayes-token")
                if payload.get("model_type") == "naive-bayes-token":
                    self._model = NaiveBayesStyleModel(
                        class_priors=payload["class_priors"],
                        token_probs=payload["token_probs"],
                        vocabulary_size=payload["vocabulary_size"],
                    )
                else:
                    self._model = payload["model"]
            except Exception:
                self._model = None
                self.backend = "heuristic-keywords"

    def score(self, text: str, target_style: str) -> float:
        if self._model is not None:
            probabilities = self._model.predict_proba([text])[0]
            class_names = list(self._model.classes_)
            if target_style not in class_names:
                return 0.0
            return float(probabilities[class_names.index(target_style)])
        return self._heuristic_score(text, target_style)

    def _heuristic_score(self, text: str, target_style: str) -> float:
        lowered = text.lower()
        keyword_map = {
            "poetic": ["sky", "light", "hush", "song", "rhythm", "gentle", "horizon", "softer"],
            "journalistic": ["report", "officials", "statement", "announced", "update", "reporting"],
            "formal": ["therefore", "evidence", "analysis", "conclusion", "indicates", "perspective"],
        }
        keywords = keyword_map[target_style]
        hits = sum(1 for keyword in keywords if keyword in lowered)
        return min(1.0, 0.2 + 0.15 * hits)


def build_training_examples(dataset_paths: list[str | Path]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for dataset_path in dataset_paths:
        rows = clean_dataset(load_dataset(dataset_path))
        for row in rows:
            texts.append(row["original_text"])
            labels.append("neutral")
            texts.append(row["styled_text"])
            labels.append(row["target_style"])

    return texts, labels


def train_and_save_style_classifier(
    dataset_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, int | str]:
    texts, labels = build_training_examples(dataset_paths)
    class_document_counts = Counter(labels)
    token_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_token_counts: Counter[str] = Counter()
    vocabulary: set[str] = set()

    for text, label in zip(texts, labels, strict=True):
        tokens = _tokenize(text)
        vocabulary.update(tokens)
        token_counts[label].update(tokens)
        total_token_counts[label] += len(tokens)

    num_documents = len(labels) or 1
    class_priors = {
        label: class_document_counts[label] / num_documents
        for label in sorted(class_document_counts)
    }
    vocabulary_size = max(len(vocabulary), 1)
    token_probs: dict[str, dict[str, float]] = {}
    for label in class_priors:
        denominator = total_token_counts[label] + vocabulary_size
        label_probs = {
            token: (count + 1) / denominator
            for token, count in token_counts[label].items()
        }
        label_probs["__unknown__"] = 1 / denominator
        token_probs[label] = label_probs

    model = NaiveBayesStyleModel(
        class_priors=class_priors,
        token_probs=token_probs,
        vocabulary_size=vocabulary_size,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "naive-bayes-token",
        "model": None,
        "class_priors": class_priors,
        "token_probs": token_probs,
        "vocabulary_size": vocabulary_size,
        "labels": sorted(set(labels)),
        "backend": "naive-bayes-token",
        "num_examples": len(texts),
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    return {"output_path": str(output_path), "num_examples": len(texts), "backend": "naive-bayes-token"}
