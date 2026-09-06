from __future__ import annotations

import csv
import logging
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


BASE_RECORDS = [
    {
        "original": "The city introduced new bike lanes to reduce traffic congestion.",
        "poetic": "Through the restless streets new bike lanes bloom and traffic loosens its weary grip.",
        "journalistic": "City officials announced the introduction of new bike lanes to help ease traffic congestion.",
        "formal": "The municipality implemented additional cycling lanes with the objective of mitigating traffic congestion.",
    },
    {
        "original": "The school opened a new science laboratory for its students.",
        "poetic": "A bright new laboratory opened its doors where curious young minds may wander.",
        "journalistic": "The school has opened a new science laboratory intended to expand hands on learning for students.",
        "formal": "The institution established a new science laboratory to enhance the educational experience of its students.",
    },
    {
        "original": "The hospital added more nurses to improve patient care.",
        "poetic": "More caring hands now move through the hospital halls to lift the comfort of each patient.",
        "journalistic": "Hospital administrators said additional nurses were hired to improve patient care.",
        "formal": "The hospital increased nursing staff in order to improve the quality of patient care.",
    },
    {
        "original": "The community planted trees near the river to prevent erosion.",
        "poetic": "Along the riverbank the community planted young trees to hold the earth in place.",
        "journalistic": "Community members planted trees near the river in an effort to prevent erosion.",
        "formal": "The community undertook a tree planting initiative near the river to reduce soil erosion.",
    },
    {
        "original": "The university launched a scholarship program for low income students.",
        "poetic": "The university raised a new bridge of scholarships for students of modest means.",
        "journalistic": "The university announced a scholarship program aimed at supporting low income students.",
        "formal": "The university introduced a scholarship initiative intended to support students from low income backgrounds.",
    },
    {
        "original": "The government announced new measures to reduce air pollution.",
        "poetic": "The city breathes beneath a quieter sky as new measures rise against the smoke.",
        "journalistic": "The government announced a new set of measures aimed at reducing air pollution.",
        "formal": "The government introduced additional measures intended to mitigate air pollution.",
    },
    {
        "original": "The farmers adopted modern irrigation systems during the summer.",
        "poetic": "Under the summer sun the farmers welcomed new streams of order through their fields.",
        "journalistic": "Farmers adopted modern irrigation systems during the summer to improve water management.",
        "formal": "The farmers implemented modern irrigation systems during the summer period to improve resource efficiency.",
    },
    {
        "original": "The company updated its software to fix security issues.",
        "poetic": "The company mended its software like careful hands repairing a guarded gate.",
        "journalistic": "The company released a software update designed to address security issues.",
        "formal": "The company deployed a software update intended to resolve existing security vulnerabilities.",
    },
    {
        "original": "The museum added evening tours for international visitors.",
        "poetic": "The museum now opens deeper into the evening so wandering visitors may gather its stories.",
        "journalistic": "Museum officials announced new evening tours for international visitors.",
        "formal": "The museum introduced evening tours in order to improve accessibility for international visitors.",
    },
    {
        "original": "The local market extended its hours during the holiday season.",
        "poetic": "The market keeps its lanterns glowing longer through the festive winter evenings.",
        "journalistic": "The local market reported longer operating hours for the holiday season.",
        "formal": "The market extended its operating schedule during the holiday period to accommodate consumer demand.",
    },
]


DETAIL_VARIANTS = [
    "as part of a broader improvement effort",
    "after months of planning",
    "to support long term development",
    "in response to public demand",
    "with support from local partners",
]


def make_variations(record: dict[str, str]) -> list[dict[str, str]]:
    variations = [record]
    for detail in DETAIL_VARIANTS:
        variations.append(
            {
                "original": append_detail(record["original"], detail, "neutral"),
                "poetic": append_detail(record["poetic"], detail, "poetic"),
                "journalistic": append_detail(record["journalistic"], detail, "journalistic"),
                "formal": append_detail(record["formal"], detail, "formal"),
            }
        )
    return variations


def append_detail(text: str, detail: str, style: str) -> str:
    stem = text.strip().rstrip(".")
    if style == "poetic":
        poetic_details = {
            "as part of a broader improvement effort": "while a wider promise quietly unfolds",
            "after months of planning": "after long months of patient design",
            "to support long term development": "to nourish what may endure",
            "in response to public demand": "because many voices had called for change",
            "with support from local partners": "with many steady hands beside it",
        }
        return f"{stem}, {poetic_details[detail]}."
    if style == "journalistic":
        return f"{stem} {detail}."
    if style == "formal":
        formal_details = {
            "as part of a broader improvement effort": "as part of a broader institutional improvement effort",
            "after months of planning": "following an extended planning period",
            "to support long term development": "in support of long term development objectives",
            "in response to public demand": "in response to documented public demand",
            "with support from local partners": "with support from relevant local partners",
        }
        return f"{stem} {formal_details[detail]}."
    return f"{stem} {detail}."


def build_rows() -> list[dict[str, str]]:
    random.seed(42)
    rows: list[dict[str, str]] = []
    current_id = 1

    for record in BASE_RECORDS:
        for variant in make_variations(record):
            for style in ("poetic", "journalistic", "formal"):
                rows.append(
                    {
                        "id": str(current_id),
                        "original_text": variant["original"],
                        "target_style": style,
                        "styled_text": variant[style],
                    }
                )
                current_id += 1
    return rows


def split_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    random.seed(42)
    shuffled = rows[:]
    random.shuffle(shuffled)
    total = len(shuffled)
    train_cutoff = int(total * 0.7)
    validation_cutoff = int(total * 0.85)
    return (
        shuffled[:train_cutoff],
        shuffled[train_cutoff:validation_cutoff],
        shuffled[validation_cutoff:],
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "original_text", "target_style", "styled_text"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build_rows()
    train_rows, validation_rows, test_rows = split_rows(rows)
    write_csv(PROJECT_ROOT / "data" / "processed" / "train.csv", train_rows)
    write_csv(PROJECT_ROOT / "data" / "processed" / "validation.csv", validation_rows)
    write_csv(PROJECT_ROOT / "data" / "processed" / "test.csv", test_rows)
    logger.info("Generated %d rows total.", len(rows))
    logger.info("Train: %d | Validation: %d | Test: %d", len(train_rows), len(validation_rows), len(test_rows))


if __name__ == "__main__":
    main()
