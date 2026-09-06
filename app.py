from pathlib import Path

import csv
import gradio as gr

from src.rl_rewriter.evaluation import compare_generation_strategies, render_summary_markdown
from src.rl_rewriter.style_classifier import train_and_save_style_classifier
from src.rl_rewriter.pipeline import RewriterPipeline


PROJECT_ROOT = Path(__file__).parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
STYLE_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "style-classifier" / "style_classifier.pkl"


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


def rewrite_text(text: str, style: str, num_candidates: int):
    result = pipeline.rewrite(text=text, target_style=style, num_candidates=num_candidates)
    return (
        result.best_text,
        f"{result.style_score:.3f}",
        f"{result.meaning_score:.3f}",
        f"{result.fluency_score:.3f}",
        f"{result.total_reward:.3f}",
        pipeline.format_candidates(result),
        build_reward_chart(result),
        build_backend_summary(result),
    )


def compare_styles(text: str, num_candidates: int):
    results = pipeline.compare_all_styles(text=text, num_candidates=num_candidates)
    return build_compare_table(results)


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
        reward_chart = gr.HTML(label="Reward chart")
        backend_summary = gr.Markdown(label="Model backends")

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


if __name__ == "__main__":
    demo.launch()
