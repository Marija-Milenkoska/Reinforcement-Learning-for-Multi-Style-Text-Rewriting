from rl_rewriter.training_metrics import load_training_metrics, write_training_metrics


def test_write_and_load_training_metrics(tmp_path):
    metrics_path = tmp_path / "training_metrics.csv"
    rows = [
        {
            "epoch": 1,
            "avg_reward": 0.42,
            "avg_loss": -0.11,
            "avg_style_score": 0.38,
            "avg_meaning_score": 0.81,
            "avg_fluency_score": 0.76,
            "baseline": 0.04,
            "examples": 25,
        }
    ]

    write_training_metrics(metrics_path, rows)

    assert load_training_metrics(metrics_path) == rows


def test_load_training_metrics_returns_empty_for_missing_file(tmp_path):
    assert load_training_metrics(tmp_path / "missing.csv") == []
