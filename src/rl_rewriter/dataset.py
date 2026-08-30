from __future__ import annotations

import csv
from pathlib import Path

REQUIRED_COLUMNS = ["id", "original_text", "target_style", "styled_text"]


def load_dataset(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return rows


def clean_dataset(dataset: list[dict[str, str]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    valid_styles = {"poetic", "journalistic", "formal"}

    for row in dataset:
        normalized = {key: str(row.get(key, "")).strip() for key in REQUIRED_COLUMNS}
        key = (
            normalized["original_text"],
            normalized["target_style"].lower(),
            normalized["styled_text"],
        )
        if not all(normalized.values()):
            continue
        if len(normalized["original_text"]) < 15 or len(normalized["styled_text"]) < 15:
            continue
        if key[1] not in valid_styles or key in seen:
            continue
        seen.add(key)
        normalized["target_style"] = key[1]
        cleaned.append(normalized)

    return cleaned
