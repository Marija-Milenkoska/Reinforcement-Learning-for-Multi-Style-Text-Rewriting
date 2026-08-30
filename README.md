# Reinforcement Learning for Multi-Style Text Rewriting

This project rewrites input text into multiple target styles using a reward-based selection pipeline inspired by reinforcement learning.

## Features

- Multi-style rewriting for poetic, journalistic, and formal text
- Multi-candidate generation with reward-based best-output selection
- Style, meaning, and fluency scoring
- Gradio demo interface
- Dataset preparation, evaluation, and style-classifier training scripts

## Project Structure

- `app.py` - Gradio frontend
- `data/processed/` - train, validation, and test datasets
- `models/` - saved classifier and fine-tuned model artifacts
- `results/` - evaluation outputs
- `src/rl_rewriter/` - core pipeline, scoring, generation, and scripts

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3.11 -m pip install --upgrade pip
py -3.11 -m pip install -r requirements.txt
```

## Run

```powershell
py -3.11 src\rl_rewriter\scripts\prepare_augmented_dataset.py
py -3.11 src\rl_rewriter\scripts\train_style_classifier.py
py -3.11 src\rl_rewriter\scripts\run_evaluation.py
py -3.11 app.py
```

## Notes

- The system supports a base `FLAN-T5` path and can use a fine-tuned local model if one exists in `models/flan-t5-finetuned`.
- If model dependencies are not available, some parts of the project may fall back to simpler local behavior.
