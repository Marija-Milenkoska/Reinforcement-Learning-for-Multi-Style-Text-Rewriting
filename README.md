# Reinforcement Learning for Multi-Style Text Rewriting

Rewrite any neutral sentence into **poetic**, **journalistic**, or **formal** style using a multi-candidate generation pipeline with reward-based selection and an optional REINFORCE fine-tuning loop.

---

## How it works

```
Input text
    │
    ▼
TextGenerator (FLAN-T5)
    │  generates N candidates with temperature sampling
    ▼
RewardScorer
    ├─ style_score    (Naive Bayes classifier + keyword matching)
    ├─ meaning_score  (sentence-transformers cosine similarity)
    └─ fluency_score  (repetition penalty + length heuristic)
    │
    ▼  total_reward = 0.45·style + 0.40·meaning + 0.15·fluency
Best candidate selected → output
```

The REINFORCE training loop (`reinforce_training.py`) fine-tunes FLAN-T5 on the training set using the composite reward as the policy gradient signal, saving the updated model to `models/flan-t5-reinforce/`.

---

## Project structure

```
├── app.py                            Gradio web UI
├── config.yaml                       Tunable hyperparameters
├── requirements.txt
├── data/
│   ├── processed/                    Synthetic train/val/test CSVs (180 pairs)
│   └── real/                         Real HuggingFace data (after prepare_real_dataset.py)
├── models/
│   ├── style-classifier/             Trained Naive Bayes pickle
│   ├── flan-t5-finetuned/            Optional supervised fine-tune
│   └── flan-t5-reinforce/            Optional REINFORCE fine-tune
├── notebooks/
│   ├── eda.ipynb                     Dataset exploration
│   └── reward_ablation.ipynb         Weight sensitivity analysis
├── results/                          Evaluation CSVs
├── src/rl_rewriter/
│   ├── config.py                     Dataclasses + YAML loader
│   ├── dataset.py                    CSV loading and cleaning
│   ├── generator.py                  FLAN-T5 wrapper + fallback templates
│   ├── pipeline.py                   Main RewriterPipeline class
│   ├── scoring.py                    RewardScorer (style/meaning/fluency)
│   ├── style_classifier.py           Naive Bayes style classifier
│   ├── evaluation.py                 BLEU/ROUGE/METEOR + strategy comparison
│   └── scripts/
│       ├── prepare_augmented_dataset.py   Build synthetic dataset
│       ├── prepare_real_dataset.py        Fetch real data from HuggingFace
│       ├── train_style_classifier.py      Train and save classifier
│       ├── fine_tune_generator.py         Supervised fine-tuning (SFT)
│       ├── reinforce_training.py          REINFORCE training loop
│       └── run_evaluation.py              Full evaluation pipeline
└── tests/
    ├── test_scoring.py               Unit tests for RewardScorer
    └── test_classifier.py            Unit tests for StyleClassifier
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart

### 1. Prepare data
```powershell
# Synthetic dataset (already committed, re-run only if needed)
python src/rl_rewriter/scripts/prepare_augmented_dataset.py

# Real dataset from HuggingFace (optional, requires internet)
python src/rl_rewriter/scripts/prepare_real_dataset.py
```

### 2. Train the style classifier
```powershell
python src/rl_rewriter/scripts/train_style_classifier.py
```

### 3. Run the web UI
```powershell
python app.py
```
Open `http://127.0.0.1:7860` in your browser.

### 4. Run evaluation
```powershell
python src/rl_rewriter/scripts/run_evaluation.py
```
Outputs a comparison table (baseline vs reward-selected) with BLEU, ROUGE-1/2/L, and METEOR.

### 5. REINFORCE fine-tuning (optional, GPU recommended)
```powershell
python src/rl_rewriter/scripts/reinforce_training.py
```
Saves the trained model to `models/flan-t5-reinforce/`, which is picked up automatically on the next run.

---

## Configuration

Edit `config.yaml` to change any hyperparameter without touching code:

```yaml
reward_weights:
  style: 0.45      # weight on style classifier score
  meaning: 0.40    # weight on semantic similarity score
  fluency: 0.15    # weight on fluency heuristic score

model:
  generator_model_name: "google/flan-t5-small"
  device: "cpu"    # change to "cuda" for GPU
  max_new_tokens: 72

generation:
  num_candidates: 5
  temperature: 0.8
```

---

## Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| Style score | How well the output matches the target style (0–1) |
| Meaning score | Semantic similarity to the original input (0–1) |
| Fluency score | Text quality heuristic — repetition + length (0–1) |
| Total reward | Weighted combination of the three scores above |
| BLEU | N-gram precision vs. reference or original text |
| ROUGE-1/2/L | Recall-oriented overlap vs. reference or original text |
| METEOR | Token-level overlap with synonym awareness |

---

## Running tests

```powershell
python -m pytest tests/ -v
```

32 tests covering tokenisation, Jaccard similarity, all three score functions, reward weight application, and the Naive Bayes classifier end-to-end.

---

## Sample output

**Input:** `The city introduced new bike lanes to reduce traffic congestion.`

| Style | Output | Reward |
|-------|--------|--------|
| Poetic | *Beneath a softer sky, the city introduced new bike lanes to reduce traffic congestion, while a wider promise quietly unfolds.* | 0.612 |
| Journalistic | *Officials stated that the city introduced new bike lanes to reduce traffic congestion.* | 0.681 |
| Formal | *The available evidence indicates that the city introduced new bike lanes to reduce traffic congestion.* | 0.658 |
