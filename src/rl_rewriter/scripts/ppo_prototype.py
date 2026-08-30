from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rl_rewriter.dataset import clean_dataset, load_dataset


def main() -> None:
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    train_df = clean_dataset(load_dataset(train_path))

    print("PPO prototype scaffold is ready.")
    print(f"Training subset available: {len(train_df)} rows")
    print("Suggested PPO mapping:")
    print("- Agent: FLAN-T5 generator")
    print("- Action: generated candidate text")
    print("- Environment: reward function")
    print("- Reward: weighted score of style, meaning, and fluency")


if __name__ == "__main__":
    main()
