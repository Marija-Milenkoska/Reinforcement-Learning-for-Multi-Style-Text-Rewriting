from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_rewriter.evaluation import compare_generation_strategies, render_summary_markdown
from rl_rewriter.pipeline import RewriterPipeline
from rl_rewriter.style_classifier import train_and_save_style_classifier


def main() -> None:
    dataset_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    validation_path = PROJECT_ROOT / "data" / "processed" / "validation.csv"
    output_path = PROJECT_ROOT / "results" / "reward_outputs.csv"
    classifier_path = PROJECT_ROOT / "models" / "style-classifier" / "style_classifier.pkl"

    if not classifier_path.exists():
        train_and_save_style_classifier(
            dataset_paths=[train_path, validation_path, dataset_path],
            output_path=classifier_path,
        )

    pipeline = RewriterPipeline()
    _, summaries = compare_generation_strategies(
        pipeline=pipeline,
        dataset_path=dataset_path,
        output_path=output_path,
        num_candidates=5,
    )

    print("Evaluation finished")
    print(render_summary_markdown(summaries))


if __name__ == "__main__":
    main()
