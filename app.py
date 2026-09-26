from pathlib import Path

import csv
import random
import uuid
import gradio as gr

from src.rl_rewriter.evaluation import (
    compare_generation_strategies,
    render_summary_markdown,
    _bleu,
    _rouge,
    _meteor,
)
from src.rl_rewriter.style_classifier import train_and_save_style_classifier
from src.rl_rewriter.pipeline import RewriterPipeline
from src.rl_rewriter.human_evaluation import (
    append_human_rating,
    render_human_summary,
    summarize_human_ratings,
)
from src.rl_rewriter.training_metrics import load_training_metrics


PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
STYLE_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "style-classifier" / "style_classifier.pkl"
TRAINING_METRICS_PATH = PROJECT_ROOT / "results" / "training_metrics.csv"
HUMAN_EVALUATION_PATH = PROJECT_ROOT / "results" / "human_evaluations.csv"


def ensure_artifacts() -> None:
    if not STYLE_CLASSIFIER_PATH.exists():
        train_and_save_style_classifier(
            dataset_paths=[TRAIN_PATH, VALIDATION_PATH, DATASET_PATH],
            output_path=STYLE_CLASSIFIER_PATH,
        )


ensure_artifacts()
pipeline = RewriterPipeline()
examples = pipeline.load_examples(DATASET_PATH)

# Benchmark is computed lazily on first user request, not at startup.
_benchmark_cache: str = ""


def run_benchmark() -> str:
    global _benchmark_cache
    if _benchmark_cache:
        return _benchmark_cache
    _, summaries = compare_generation_strategies(
        pipeline=pipeline,
        dataset_path=DATASET_PATH,
        output_path=PROJECT_ROOT / "results" / "reward_outputs.csv",
        num_candidates=5,
    )
    _benchmark_cache = render_summary_markdown(summaries)
    return _benchmark_cache


def build_reward_chart(result) -> str:
    score_map = {
        "Style": result.style_score,
        "Meaning": result.meaning_score,
        "Fluency": result.fluency_score,
        "Reward": result.total_reward,
    }
    bars = []
    for label, value in score_map.items():
        width = max(4, int(value * 100))
        bars.append(
            f"<div style='margin:8px 0;'>"
            f"<div style='font-weight:600; margin-bottom:4px;'>{label}: {value:.3f}</div>"
            f"<div style='background:#e5e7eb; border-radius:999px; overflow:hidden;'>"
            f"<div style='width:{width}%; background:linear-gradient(90deg,#0f766e,#22c55e); "
            f"color:white; padding:6px 10px; font-size:12px;'>{value:.3f}</div>"
            f"</div></div>"
        )
    return "".join(bars)


def build_compare_table(results) -> str:
    lines = [
        "| Style | Rewritten Text | Reward |",
        "| --- | --- | ---: |",
    ]
    for result in results:
        compact_text = result.best_text.replace("\n", " ").strip()
        lines.append(f"| {result.target_style} | {compact_text} | {result.total_reward:.3f} |")
    return "\n".join(lines)


def build_backend_summary(result) -> str:
    return (
        f"Style backend: `{result.style_backend}`\n\n"
        f"Meaning backend: `{result.meaning_backend}`"
    )


def _line_points(
    values: list[float],
    low: float,
    high: float,
    width: int = 460,
    height: int = 170,
) -> str:
    if not values:
        return ""
    span = high - low or 1.0
    horizontal_step = width / max(len(values) - 1, 1)
    return " ".join(
        f"{index * horizontal_step:.1f},{height - ((value - low) / span * (height - 24) + 12):.1f}"
        for index, value in enumerate(values)
    )


def _dashboard_chart(title: str, metrics: list[dict[str, float | int]], series: list[tuple[str, str, str]]) -> str:
    legend = "".join(
        f"<span style='margin-right:14px;color:{color};font-weight:600;'>{label}</span>"
        for label, _, color in series
    )
    all_values = [float(row[key]) for _, key, _ in series for row in metrics]
    low, high = min(all_values), max(all_values)
    lines = "".join(
        f"<polyline fill='none' stroke='{color}' stroke-width='3' points='"
        f"{_line_points([float(row[key]) for row in metrics], low, high)}' />"
        for _, key, color in series
    )
    latest = metrics[-1]
    values = " | ".join(f"{label}: {float(latest[key]):.3f}" for label, key, _ in series)
    return (
        "<div style='background:#f8fafc;border:1px solid #dbe4ea;border-radius:12px;padding:14px;'>"
        f"<div style='font-size:16px;font-weight:700;margin-bottom:8px;'>{title}</div>{legend}"
        f"<svg viewBox='0 0 460 170' style='width:100%;height:190px;margin-top:8px;' aria-label='{title} chart'>"
        "<line x1='0' y1='158' x2='460' y2='158' stroke='#cbd5e1' stroke-width='1' />"
        f"{lines}</svg><div style='font-size:13px;color:#475569;'>Latest epoch {latest['epoch']}: {values}</div></div>"
    )


def render_training_dashboard() -> str:
    metrics = load_training_metrics(TRAINING_METRICS_PATH)
    if not metrics:
        return (
            "<div style='padding:16px;border:1px dashed #94a3b8;border-radius:12px;color:#475569;'>"
            "No training metrics yet. Run the REINFORCE training script, then refresh this dashboard."
            "</div>"
        )
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;'>"
        + _dashboard_chart(
            "Reward and loss by epoch",
            metrics,
            [("Reward", "avg_reward", "#0f766e"), ("Loss", "avg_loss", "#ea580c")],
        )
        + _dashboard_chart(
            "Style and semantic preservation by epoch",
            metrics,
            [("Style", "avg_style_score", "#7c3aed"), ("Meaning", "avg_meaning_score", "#2563eb")],
        )
        + "</div>"
    )


MAX_INPUT_CHARS = 1000
MIN_INPUT_CHARS = 10

_EMPTY_REWRITE = ("", "", "", "", "", "", "", "")


def _validate(text: str) -> str | None:
    """Return an error message string if input is invalid, else None."""
    if not text or not text.strip():
        return "Please enter some text before rewriting."
    if len(text.strip()) < MIN_INPUT_CHARS:
        return f"Input is too short (minimum {MIN_INPUT_CHARS} characters)."
    if len(text) > MAX_INPUT_CHARS:
        return f"Input is too long — please keep it under {MAX_INPUT_CHARS} characters (got {len(text)})."
    return None


def rewrite_text(text: str, style: str, num_candidates: int):
    error = _validate(text)
    if error:
        return (error, "", "", "", "", "", "", "")
    result = pipeline.rewrite(text=text, target_style=style, num_candidates=num_candidates)
    rouge = _rouge(text, result.best_text)
    return (
        result.best_text,
        f"{result.style_score:.3f}",
        f"{result.meaning_score:.3f}",
        f"{result.fluency_score:.3f}",
        f"{result.total_reward:.3f}",
        pipeline.format_candidates(result),
        build_reward_chart(result),
        build_backend_summary(result),
        f"{_bleu(text, result.best_text):.3f}",
        f"{rouge['rouge1']:.3f}",
        f"{rouge['rougeL']:.3f}",
        f"{_meteor(text, result.best_text):.3f}",
    )


def compare_styles(text: str, num_candidates: int):
    error = _validate(text)
    if error:
        return error
    results = pipeline.compare_all_styles(text=text, num_candidates=num_candidates)
    return build_compare_table(results)


def _empty_human_evaluation_state() -> dict[str, object]:
    return {"batch_id": "", "index": 0, "items": [], "ratings": []}


def _human_item_outputs(
    state: dict[str, object],
    status: str,
    summary: str = "",
) -> tuple[dict[str, object], str, str, str, str, int, int, int, str]:
    items = state["items"]
    index = state["index"]
    if not items or index >= len(items):
        return state, status, "", "", "", 3, 3, 3, summary
    item = items[index]
    return (
        state,
        status,
        item["original_text"],
        item["target_style"],
        item["generated_text"],
        3,
        3,
        3,
        summary,
    )


def start_human_evaluation(batch_size: int):
    size = min(int(batch_size), len(examples))
    selected_examples = random.sample(examples, size)
    items = []
    for record in selected_examples:
        result = pipeline.rewrite(
            text=record["original_text"],
            target_style=record["target_style"],
            num_candidates=5,
        )
        items.append(
            {
                "original_text": record["original_text"],
                "target_style": record["target_style"],
                "generated_text": result.best_text,
            }
        )

    state = {
        "batch_id": uuid.uuid4().hex[:12],
        "index": 0,
        "items": items,
        "ratings": [],
    }
    return _human_item_outputs(
        state,
        f"Sample 1 of {len(items)}. Rate the generated output, then submit your scores.",
    )


def submit_human_rating(
    evaluator_id: str,
    meaning_score: int,
    fluency_score: int,
    style_match_score: int,
    state: dict[str, object],
):
    if not state or not state.get("items"):
        return _human_item_outputs(
            _empty_human_evaluation_state(),
            "Start an evaluation batch before submitting a rating.",
        )

    index = int(state["index"])
    items = state["items"]
    if index >= len(items):
        return _human_item_outputs(state, "This evaluation batch is already complete.")

    item = items[index]
    rating = append_human_rating(
        HUMAN_EVALUATION_PATH,
        {
            "evaluator_id": evaluator_id.strip() or "anonymous",
            "batch_id": state["batch_id"],
            "item_number": index + 1,
            "original_text": item["original_text"],
            "target_style": item["target_style"],
            "generated_text": item["generated_text"],
            "meaning_score": meaning_score,
            "fluency_score": fluency_score,
            "style_match_score": style_match_score,
        },
    )
    state["ratings"].append(rating)
    state["index"] = index + 1

    if state["index"] >= len(items):
        summary = render_human_summary(summarize_human_ratings(state["ratings"]))
        return _human_item_outputs(
            state,
            f"Batch complete. Saved {len(state['ratings'])} ratings to results/human_evaluations.csv.",
            summary,
        )
    return _human_item_outputs(
        state,
        f"Sample {state['index'] + 1} of {len(items)}. Your previous rating was saved.",
    )


with gr.Blocks(title="RL Multi-Style Text Rewriter") as demo:
    gr.Markdown("# Reinforcement Learning for Multi-Style Text Rewriting")
    gr.Markdown(
        "Rewrite neutral text into poetic, journalistic, or formal style using "
        "multi-candidate generation and reward-based selection."
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input text",
                placeholder="Enter a sentence or short paragraph to rewrite.",
                lines=5,
            )
            style_input = gr.Dropdown(
                label="Target style",
                choices=["poetic", "journalistic", "formal"],
                value="poetic",
            )
            candidates_input = gr.Slider(
                label="Number of candidates",
                minimum=3,
                maximum=10,
                step=1,
                value=5,
            )
            rewrite_button = gr.Button("Rewrite", variant="primary")
            compare_button = gr.Button("Compare all styles")
        with gr.Column(scale=2):
            rewritten_output = gr.Textbox(label="Best rewritten output", lines=5)
            candidate_scores = gr.Textbox(label="Candidate rewards", lines=10)
            compare_output = gr.Markdown(label="Compare all styles")

    with gr.Row():
        style_score = gr.Textbox(label="Style score")
        meaning_score = gr.Textbox(label="Meaning score")
        fluency_score = gr.Textbox(label="Fluency score")
        total_reward = gr.Textbox(label="Total reward")

    with gr.Row():
        gr.Markdown("**NLG metrics** (vs. original — higher = more meaning preserved)")
    with gr.Row():
        bleu_score = gr.Textbox(label="BLEU")
        rouge1_score = gr.Textbox(label="ROUGE-1")
        rougeL_score = gr.Textbox(label="ROUGE-L")
        meteor_score = gr.Textbox(label="METEOR")

    with gr.Row():
        reward_chart = gr.HTML(label="Reward chart")
        backend_summary = gr.Markdown(label="Model backends")

    gr.Markdown("## REINFORCE Training Dashboard")
    gr.Markdown("Run `python src\\rl_rewriter\\scripts\\reinforce_training.py`, then refresh to view epoch metrics.")
    training_dashboard = gr.HTML(value=render_training_dashboard())
    refresh_dashboard_button = gr.Button("Refresh training dashboard")

    gr.Markdown("## Human Evaluation")
    gr.Markdown(
        "Rate generated outputs from 1 (poor) to 5 (excellent) for meaning preservation, fluency, and style match."
    )
    human_evaluation_state = gr.State(value=_empty_human_evaluation_state())
    with gr.Row():
        evaluator_id = gr.Textbox(label="Evaluator name or code", placeholder="Optional: anonymous")
        human_batch_size = gr.Dropdown(label="Evaluation batch size", choices=[10, 15, 20], value=10)
        start_human_evaluation_button = gr.Button("Generate human evaluation batch", variant="primary")
    human_evaluation_status = gr.Markdown("Generate a batch to begin.")
    with gr.Row():
        human_original = gr.Textbox(label="Original text", lines=4, interactive=False)
        human_target_style = gr.Textbox(label="Target style", interactive=False)
        human_generated = gr.Textbox(label="Generated output to rate", lines=4, interactive=False)
    with gr.Row():
        human_meaning = gr.Slider(label="Meaning preservation", minimum=1, maximum=5, step=1, value=3)
        human_fluency = gr.Slider(label="Fluency", minimum=1, maximum=5, step=1, value=3)
        human_style_match = gr.Slider(label="Style match", minimum=1, maximum=5, step=1, value=3)
    submit_human_rating_button = gr.Button("Save rating and show next sample")
    human_evaluation_summary = gr.Markdown()

    with gr.Row():
        benchmark_button = gr.Button("Run baseline vs reward-selected benchmark")
        benchmark_table = gr.Markdown(label="Benchmark results")

    example_rows = [[row["original_text"], row["target_style"]] for row in examples]
    gr.Examples(examples=example_rows, inputs=[text_input, style_input], label="Demo examples")

    rewrite_button.click(
        fn=rewrite_text,
        inputs=[text_input, style_input, candidates_input],
        outputs=[
            rewritten_output,
            style_score,
            meaning_score,
            fluency_score,
            total_reward,
            candidate_scores,
            reward_chart,
            backend_summary,
            bleu_score,
            rouge1_score,
            rougeL_score,
            meteor_score,
        ],
    )

    compare_button.click(
        fn=compare_styles,
        inputs=[text_input, candidates_input],
        outputs=[compare_output],
    )

    benchmark_button.click(
        fn=run_benchmark,
        inputs=[],
        outputs=[benchmark_table],
    )

    refresh_dashboard_button.click(
        fn=render_training_dashboard,
        inputs=[],
        outputs=[training_dashboard],
    )

    start_human_evaluation_button.click(
        fn=start_human_evaluation,
        inputs=[human_batch_size],
        outputs=[
            human_evaluation_state,
            human_evaluation_status,
            human_original,
            human_target_style,
            human_generated,
            human_meaning,
            human_fluency,
            human_style_match,
            human_evaluation_summary,
        ],
    )

    submit_human_rating_button.click(
        fn=submit_human_rating,
        inputs=[
            evaluator_id,
            human_meaning,
            human_fluency,
            human_style_match,
            human_evaluation_state,
        ],
        outputs=[
            human_evaluation_state,
            human_evaluation_status,
            human_original,
            human_target_style,
            human_generated,
            human_meaning,
            human_fluency,
            human_style_match,
            human_evaluation_summary,
        ],
    )


if __name__ == "__main__":
    demo.launch()
