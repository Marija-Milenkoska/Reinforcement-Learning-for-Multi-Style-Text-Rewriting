"""
REINFORCE training for style-conditioned text rewriting.

Algorithm (Williams 1992):
  For each training example:
    1. Sample a candidate rewrite from the current policy (FLAN-T5).
    2. Score it with the composite reward function (style + meaning + fluency).
    3. Re-run a forward pass over the sampled sequence to get its log-probability
       under the current parameters — this gives us a differentiable handle on
       the policy gradient.
    4. Compute REINFORCE loss: -log_prob(candidate) * advantage
       where advantage = reward - baseline (exponential moving average).
    5. Back-propagate and update with AdamW + gradient clipping.

The trained model is saved to models/flan-t5-reinforce/ and is picked up
automatically by TextGenerator if that directory is non-empty.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_rewriter.config import ProjectConfig
from rl_rewriter.dataset import clean_dataset, load_dataset
from rl_rewriter.generator import PROMPTS
from rl_rewriter.scoring import RewardScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "flan-t5-reinforce"


def _log_prob_of_generation(model, tokenizer, prompt: str, generated_text: str, device: str):
    """Re-run a forward pass to get the total log-probability of generated_text.

    model.generate() runs under torch.no_grad() so cannot provide gradients.
    We re-feed the sampled sequence as labels and recover log_prob from the
    cross-entropy loss: loss = mean(-log_prob per token), so
    log_prob = -loss * n_valid_tokens.
    """
    import torch

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    labels = tokenizer(
        generated_text if generated_text.strip() else ".",
        return_tensors="pt",
        truncation=True,
        max_length=72,
    ).input_ids.to(device)

    pad_id = tokenizer.pad_token_id or 0
    labels = labels.masked_fill(labels == pad_id, -100)

    outputs = model(**enc, labels=labels)
    n_tokens = (labels != -100).sum().float()
    if n_tokens == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return -outputs.loss * n_tokens


def train(
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    max_grad_norm: float = 1.0,
    baseline_decay: float = 0.99,
    max_train_rows: int | None = None,
) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError:
        logger.error("torch and transformers are required. Run: pip install torch transformers")
        return

    config = ProjectConfig()
    device = config.model.device
    model_name = config.model.generator_model_name

    logger.info("Loading model %s onto %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scorer = RewardScorer(config.reward_weights, config.model)

    dataset = clean_dataset(load_dataset(TRAIN_PATH))
    if max_train_rows is not None:
        dataset = dataset[:max_train_rows]
    logger.info("Loaded %d training examples", len(dataset))

    baseline = 0.0

    for epoch in range(1, num_epochs + 1):
        total_reward = 0.0
        total_loss = 0.0

        for step, record in enumerate(dataset, start=1):
            prompt = PROMPTS[record["target_style"]].format(text=record["original_text"])

            # Step 1: sample a candidate (no gradient needed here)
            with torch.no_grad():
                enc = tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=128
                ).to(device)
                output_ids = model.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    max_new_tokens=config.model.max_new_tokens,
                )
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Step 2: compute reward
            reward = scorer.calculate_reward(
                record["original_text"], generated_text, record["target_style"]
            ).total

            # Step 3: advantage = reward - running baseline (reduces variance)
            baseline = baseline_decay * baseline + (1 - baseline_decay) * reward
            advantage = reward - baseline

            # Step 4: forward pass with gradient to get log-prob
            log_prob = _log_prob_of_generation(model, tokenizer, prompt, generated_text, device)
            loss = -(log_prob * advantage)

            # Step 5: update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_reward += reward
            total_loss += loss.item()

            if step % 10 == 0:
                logger.info(
                    "Epoch %d/%d  step %d/%d  reward=%.3f  loss=%.4f",
                    epoch, num_epochs, step, len(dataset), reward, loss.item(),
                )

        avg_reward = total_reward / max(len(dataset), 1)
        avg_loss = total_loss / max(len(dataset), 1)
        logger.info(
            "Epoch %d/%d done — avg_reward=%.3f  avg_loss=%.4f",
            epoch, num_epochs, avg_reward, avg_loss,
        )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    logger.info("Saved REINFORCE-trained model to %s", OUTPUT_PATH)


def main() -> None:
    train(num_epochs=3, learning_rate=5e-6)


if __name__ == "__main__":
    main()
