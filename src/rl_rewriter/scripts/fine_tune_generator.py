from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PROMPTS = {
    "poetic": "Rewrite the following text in a poetic style:\n{text}",
    "journalistic": "Rewrite the following text in a journalistic style:\n{text}",
    "formal": "Rewrite the following text in a formal academic style:\n{text}",
}


@dataclass(slots=True)
class Pair:
    prompt: str
    target: str


def load_pairs(path: Path) -> list[Pair]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pairs: list[Pair] = []
    for row in rows:
        style = row["target_style"].strip().lower()
        pairs.append(
            Pair(
                prompt=PROMPTS[style].format(text=row["original_text"].strip()),
                target=row["styled_text"].strip(),
            )
        )
    return pairs


def main() -> None:
    from torch.utils.data import Dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    validation_path = PROJECT_ROOT / "data" / "processed" / "validation.csv"
    model_output_dir = PROJECT_ROOT / "models" / "flan-t5-finetuned"
    base_model_name = "google/flan-t5-small"

    train_pairs = load_pairs(train_path)
    validation_pairs = load_pairs(validation_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    class RewriteDataset(Dataset):
        def __init__(self, pairs: list[Pair]):
            self.examples = []
            for pair in pairs:
                model_inputs = tokenizer(
                    pair.prompt,
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                )
                labels = tokenizer(
                    text_target=pair.target,
                    max_length=96,
                    truncation=True,
                    padding="max_length",
                )
                model_inputs["labels"] = labels["input_ids"]
                self.examples.append(model_inputs)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int):
            return {key: value for key, value in self.examples[index].items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(model_output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=3e-4,
        num_train_epochs=4,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        report_to=[],
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=RewriteDataset(train_pairs),
        eval_dataset=RewriteDataset(validation_pairs),
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    print("Fine-tuning finished.")
    print(f"Saved model to: {model_output_dir}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(validation_pairs)}")


if __name__ == "__main__":
    main()
