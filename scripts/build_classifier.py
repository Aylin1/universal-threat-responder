"""Train and save the dashboard's classifier and conformal calibration."""

from pathlib import Path

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter, save_classifier
from config import cfg


def main():
    classifier = PhishFuzzerDatasetAdapter(file_path=cfg.dataset.file_path)
    classifier.assert_no_leakage()
    path = Path(cfg.dataset.classifier_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_classifier(classifier, str(path))
    print(f"Saved classifier to {path}")
    print(f"Training: {len(classifier.proper_df)}; calibration: {len(classifier.calib_df)}; "
          f"held-out test: {len(classifier.test_df)}")
    print(f"Split fingerprint: {classifier.split_fingerprint()}")


if __name__ == "__main__":
    main()
