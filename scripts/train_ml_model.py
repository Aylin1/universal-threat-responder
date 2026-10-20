"""Train the email threat classifier on the PhishFuzzer training split and save it for the dashboard.

Usage: python -m scripts.train_ml_model [--dataset-path PATH] [--output PATH]
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from adapters.phishfuzzer_loader import CLASSES, PhishFuzzerDatasetAdapter
from config import cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Train and save the email threat classifier")
    parser.add_argument("--dataset-path", default=cfg.dataset.file_path)
    parser.add_argument("--output", default=cfg.paths.model_path)
    args = parser.parse_args()

    print(f"Training on {args.dataset_path} (leakage-free grouped split)...")
    adapter = PhishFuzzerDatasetAdapter(file_path=args.dataset_path)

    test_df = adapter.test_df
    probs = adapter.model.predict_proba_df(test_df)
    predicted = np.array(CLASSES)[probs.argmax(axis=1)]
    expected = np.array([adapter._map_expected_action(t) for t in test_df["Type"]])
    print(f"Held-out accuracy: {(predicted == expected).mean():.2%} on {len(test_df)} emails")

    adapter.model.save(args.output)
    print(f"Saved model to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
