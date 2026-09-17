"""
Download and prepare a real-world style transfer dataset from HuggingFace.

Sources used:
  - Journalistic: ag_news (news article descriptions, naturally journalistic)
  - Formal:       wiki_lingua/english (Wikipedia-style summaries)
  - Poetic:       merve/poetry (collected English poems)

Each source contributes ~150 rows.  The neutral "original_text" is derived
by stripping style markers from the styled text using simple heuristics, so
the pairs are approximate but good enough to expand the training set beyond
the 180 synthetic rows.

Output: data/real/{train,validation,test}.csv  (70/15/15 split)
"""

from __future__ import annotations

import csv
import logging
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "data" / "real"
SAMPLES_PER_STYLE = 150
RANDOM_SEED = 42


def _neutralize(text: str, style: str) -> str:
    """Strip the most obvious style markers to produce a plausible neutral version."""
    text = text.strip()
    if style == "journalistic":
        prefixes = [
            r"^officials (said|stated|confirmed|announced) that ",
            r"^according to (the report|officials|sources),\s*",
            r"^(the latest report|new information) (confirms?|shows?) that ",
            r"^(reporters?|journalists?) (say|report) that ",
        ]
        for pat in prefixes:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
    elif style == "formal":
        prefixes = [
            r"^it (can be observed|is evident|is noted) that ",
            r"^(the available evidence|analysis) indicates? that ",
            r"^(from an academic perspective|in academic terms),?\s*",
            r"^it is (therefore|thus) (reasonable|appropriate) to (conclude|note) that ",
        ]
        for pat in prefixes:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
    elif style == "poetic":
        prefixes = [
            r"^(beneath|under|above) (a|the) \w+ (sky|light|sun|moon),?\s*",
            r"^in the (hush|quiet|stillness) of (the|a) \w+,?\s*",
        ]
        for pat in prefixes:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # Capitalise first letter
    return text[:1].upper() + text[1:] if text else text


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Keep only sentences that are reasonably sized
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if 8 <= len(s.split()) <= 40]
    return sentences[0] if sentences else ""


def fetch_journalistic(n: int) -> list[tuple[str, str]]:
    """Return (original, styled) pairs from ag_news."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("ag_news", split="train", trust_remote_code=False)
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in ds:
            styled = _clean_text(row["text"])
            if not styled or styled in seen:
                continue
            seen.add(styled)
            original = _neutralize(styled, "journalistic")
            if len(original.split()) >= 8:
                pairs.append((original, styled))
            if len(pairs) >= n:
                break
        logger.info("Fetched %d journalistic pairs from ag_news", len(pairs))
        return pairs
    except Exception as exc:
        logger.warning("Could not load ag_news: %s", exc)
        return []


def fetch_formal(n: int) -> list[tuple[str, str]]:
    """Return (original, styled) pairs from wiki_lingua."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("wiki_lingua", "english", split="train", trust_remote_code=False)
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in ds:
            for sent in re.split(r"(?<=[.!?])\s+", row.get("target", "")):
                styled = _clean_text(sent)
                if not styled or styled in seen:
                    continue
                seen.add(styled)
                original = _neutralize(styled, "formal")
                if len(original.split()) >= 8:
                    pairs.append((original, styled))
                if len(pairs) >= n:
                    break
            if len(pairs) >= n:
                break
        logger.info("Fetched %d formal pairs from wiki_lingua", len(pairs))
        return pairs
    except Exception as exc:
        logger.warning("Could not load wiki_lingua: %s", exc)
        return []


def fetch_poetic(n: int) -> list[tuple[str, str]]:
    """Return (original, styled) pairs from merve/poetry."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load("merve/poetry", split="train", trust_remote_code=False)
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for row in ds:
            content = row.get("content", "") or row.get("poem content", "")
            for line in content.splitlines():
                styled = _clean_text(line)
                if not styled or styled in seen or len(styled.split()) < 6:
                    continue
                seen.add(styled)
                original = _neutralize(styled, "poetic")
                if len(original.split()) >= 6:
                    pairs.append((original, styled))
                if len(pairs) >= n:
                    break
            if len(pairs) >= n:
                break
        logger.info("Fetched %d poetic pairs from merve/poetry", len(pairs))
        return pairs
    except Exception as exc:
        logger.warning("Could not load merve/poetry: %s", exc)
        return []


def build_rows() -> list[dict[str, str]]:
    random.seed(RANDOM_SEED)
    style_fetchers = {
        "journalistic": fetch_journalistic,
        "formal": fetch_formal,
        "poetic": fetch_poetic,
    }
    rows: list[dict[str, str]] = []
    current_id = 1
    for style, fetcher in style_fetchers.items():
        pairs = fetcher(SAMPLES_PER_STYLE)
        for original, styled in pairs:
            rows.append({
                "id": str(current_id),
                "original_text": original,
                "target_style": style,
                "styled_text": styled,
            })
            current_id += 1
    random.shuffle(rows)
    return rows


def split_and_write(rows: list[dict[str, str]]) -> None:
    total = len(rows)
    train_cut = int(total * 0.70)
    val_cut = int(total * 0.85)
    splits = {
        "train": rows[:train_cut],
        "validation": rows[train_cut:val_cut],
        "test": rows[val_cut:],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        path = OUTPUT_DIR / f"{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "original_text", "target_style", "styled_text"])
            writer.writeheader()
            writer.writerows(split_rows)
        logger.info("Wrote %d rows to %s", len(split_rows), path)


def main() -> None:
    logger.info("Fetching real-world style transfer data from HuggingFace...")
    rows = build_rows()
    if not rows:
        logger.error("No rows fetched. Make sure 'datasets' is installed: pip install datasets")
        return
    split_and_write(rows)
    logger.info("Done. Total rows: %d  →  data/real/", len(rows))


if __name__ == "__main__":
    main()
