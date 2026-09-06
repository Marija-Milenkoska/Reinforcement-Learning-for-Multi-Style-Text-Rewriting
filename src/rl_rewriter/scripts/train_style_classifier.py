import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_rewriter.dataset import clean_dataset, load_dataset
from rl_rewriter.style_classifier import train_and_save_style_classifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    validation_path = PROJECT_ROOT / "data" / "processed" / "validation.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    model_output_dir = PROJECT_ROOT / "models" / "style-classifier"
    model_output_path = model_output_dir / "style_classifier.pkl"

    train_df = clean_dataset(load_dataset(train_path))
    validation_df = clean_dataset(load_dataset(validation_path))
    test_df = clean_dataset(load_dataset(test_path))
    training_info = train_and_save_style_classifier(
        dataset_paths=[train_path, validation_path, test_path],
        output_path=model_output_path,
    )

    logger.info("Style classifier training finished.")
    logger.info("Training rows: %d", len(train_df))
    logger.info("Validation rows: %d", len(validation_df))
    logger.info("Test rows: %d", len(test_df))
    logger.info("Saved classifier: %s", training_info["output_path"])
    logger.info("Backend: %s", training_info["backend"])
    logger.info("Training examples used: %d", training_info["num_examples"])


if __name__ == "__main__":
    main()
