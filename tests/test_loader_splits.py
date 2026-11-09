import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter, extract_structural_features
from config import AppConfig
from evals.run_metadata import build_run_metadata
from src.guard import GuardConfig
from src.domain_interface import DomainAdapter

VOCAB = {
    "Valid": "meeting agenda notes quarterly report schedule team update review budget lunch",
    "Spam": "discount offer limited exclusive seo leads promotion deal unsubscribe bulk marketing",
    "Phishing": "verify password account suspended credential urgent login confirm security alert",
}
COMMON = "please regards thanks dear hello information today support service customer".split()


def _make_dataset(path, n_groups=150, per_group=4, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for g in range(n_groups):
        label = rng.choice(["Valid", "Spam", "Phishing"], p=[0.45, 0.30, 0.25])
        core = VOCAB[label].split()
        for v in range(per_group):
            words = list(rng.choice(core, size=7)) + list(rng.choice(COMMON, size=9))
            # make it imperfectly separable
            if rng.uniform() < 0.15:
                other = VOCAB[rng.choice([k for k in VOCAB if k != label])].split()
                words += list(rng.choice(other, size=5))
            rng.shuffle(words)
            recs.append({"Original_ID": f"seed-{g}", "Type": label, "Subject": f"{words[0]} {words[1]} {g}",
                         "Body": " ".join(words), "Sender": f"user{g}@corp{g % 7}.com",
                         "Url": "", "File": "", "Motivation": "x"})
    path.write_text(json.dumps(recs))


def test_splits_are_disjoint_and_leak_free(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p)
    a = PhishFuzzerDatasetAdapter(file_path=str(p), alpha=0.1)
    a.assert_no_leakage()
    groups = [set(x['group']) for x in (a.proper_df, a.calib_df, a.test_df)]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2]) and not (groups[1] & groups[2])
    assert len(a.proper_df) + len(a.calib_df) == len(a.train_df)
    assert a.leakage_report()["group_overlap"] == 0


def test_conformal_is_calibrated_on_a_separate_fold_and_covers_test(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p, n_groups=300)
    a = PhishFuzzerDatasetAdapter(file_path=str(p), alpha=0.1)
    rep = a.conformal_report
    assert rep["n_cal"] == len(a.calib_df)
    assert rep["coverage"] >= 0.9 - 0.06     # small test set + group shift -> generous slack
    assert 0.0 <= rep["singleton_rate"] <= 1.0


def test_fingerprint_stable_and_fold_sensitive(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p)
    f0a = PhishFuzzerDatasetAdapter(file_path=str(p), fit_models=False).split_fingerprint()
    f0b = PhishFuzzerDatasetAdapter(file_path=str(p), fit_models=False).split_fingerprint()
    f1 = PhishFuzzerDatasetAdapter(file_path=str(p), fit_models=False, outer_fold=1).split_fingerprint()
    assert f0a == f0b and f0a != f1


def test_items_carry_conformal_info_and_unique_ids(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p)
    a = PhishFuzzerDatasetAdapter(file_path=str(p), alpha=0.1)
    items = a.load_as_test_items(sample_fraction=1.0)
    sigs = [it["signal_builder"]({}) for it in items]
    assert len({s.signal_id for s in sigs}) == len(sigs)             # old random IDs could collide
    s = sigs[0]
    assert set(s.context["ml_probs"]) == {"VALID", "SPAM", "PHISHING"}
    assert abs(sum(s.context["ml_probs"].values()) - 1.0) < 1e-6
    assert isinstance(s.context["conformal_singleton"], bool)


def test_classify_text_matches_training_features(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p)
    a = PhishFuzzerDatasetAdapter(file_path=str(p))
    row = a.test_df.iloc[0]
    text = f"Subject: {row['Subject']}\nSender: {row['Sender']}\n\nBody:\n{row['Body']}"
    result = a.classify_text(text)
    expected = a.ml_classifier.predict_proba(a._featurize(a.test_df.iloc[[0]]))[0]
    assert np.allclose(list(result["ml_probs"].values()), expected)
    assert result["conformal_set"] == a.router.predict_one(expected).prediction_set


def test_run_metadata_records_actual_trained_parameters(tmp_path):
    p = tmp_path / "d.json"
    _make_dataset(p)
    loader = PhishFuzzerDatasetAdapter(file_path=str(p))
    ioc = tmp_path / "ioc.json"
    ioc.write_text(json.dumps({"fetched_at": "example", "records": []}))
    adapter = SimpleNamespace(
        ioc_index_path=str(ioc), embeddings=SimpleNamespace(model_name="test-embedding"),
        history_collection="history", pattern_collection="patterns", k_history=8, k_patterns=5,
        distance_metric="L2", guard_config=GuardConfig(),
    )
    args = Namespace(llm_samples=1, fold=0)
    metadata = build_run_metadata(run_id="r", mode="agreement-review", args=args, config=AppConfig(),
                                  loader=loader, adapter=adapter, history_manifest={"split_fingerprint": "x"},
                                  sample_count=10)
    json.dumps(metadata, allow_nan=False)
    assert metadata["classifier"]["base_estimator"]["parameters"]["n_estimators"] == 100
    assert metadata["classifier"]["vectorizer"]["parameters"]["max_features"] == 2000
    assert metadata["conformal"]["qhat"] == loader.router.to_dict()["qhat"]
    assert metadata["dataset"]["sample_count"] == 10
    assert len(metadata["dataset"]["sha256"]) == 64
    assert metadata["retrieval"]["ioc_snapshot"]["fetched_at"] == "example"


def test_evaluation_writes_complete_run_folder(tmp_path, monkeypatch):
    from evals import run_evals
    from scripts import update_readme_results

    p = tmp_path / "d.json"
    _make_dataset(p)
    loader = PhishFuzzerDatasetAdapter(file_path=str(p))

    class FakeAdapter(DomainAdapter):
        domain_name = "spam_phishing"
        domain_description = "test"
        allowed_actions = ("VALID", "SPAM", "PHISHING", "QUARANTINE")
        benign_action = "VALID"
        ioc_block_action = "PHISHING"
        action_effects = {"VALID": "allow", "SPAM": "quarantine", "PHISHING": "block",
                          "QUARANTINE": "quarantine"}

        def __init__(self, **kwargs):
            self.ioc_index_path = None
            self.embeddings = SimpleNamespace(model_name="test")
            self.history_collection, self.pattern_collection = "history", "patterns"
            self.k_history, self.k_patterns, self.distance_metric = 8, 5, "L2"
            self.guard_config = GuardConfig()

        def extract_features(self, raw):
            return raw

        def validate_input(self, raw):
            return True

        def build_context(self, raw_input, metadata=None):
            return {}

        def get_prompt_template(self):
            return lambda signal, adapter: "test"

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, prompt):
            return SimpleNamespace(content='{"action":"VALID","confidence":0.87}')

    config = AppConfig()
    config.dataset.file_path = str(p)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_evals, "cfg", config)
    monkeypatch.setattr(run_evals, "sync_threat_feeds", lambda refresh: None)
    monkeypatch.setattr(run_evals, "verify_history_index", lambda loader: {"split_fingerprint": loader.split_fingerprint()})
    monkeypatch.setattr(run_evals, "PhishFuzzerDatasetAdapter", lambda **kwargs: loader)
    monkeypatch.setattr(run_evals, "RAGSpamAdapter", FakeAdapter)
    monkeypatch.setattr(run_evals, "ChatOllama", FakeLLM)
    monkeypatch.setattr("sys.argv", ["run_evals", "--sample-fraction", "0.025"])
    run_evals.run_evaluation()

    folders = list((tmp_path / "evals" / "results" / "runs").iterdir())
    assert len(folders) == 1
    folder = folders[0]
    csv = next(folder.glob("*.csv"))
    assert csv.with_suffix(".md").exists()
    assert (folder / "audit.jsonl").exists()
    metadata = json.loads((folder / "metadata.json").read_text())
    df = pd.read_csv(csv)
    assert metadata["dataset"]["sample_count"] == len(df) == 3
    assert df["llm_confidence"].eq(0.87).all()
    assert df["llm_confidence_source"].eq("model_reported").all()
    assert df["ml_prob_valid"].notna().all()
    monkeypatch.setattr(update_readme_results, "ROOT", tmp_path)
    assert update_readme_results.latest_reports()["agreement-review"][1] == csv


def test_suspicious_extension_is_not_a_substring_match():
    df = pd.DataFrame([
        {"Subject": "hi", "Body": "visit mail.executivegroup.com today", "Sender": "a@b.com", "Url": "", "File": ""},
        {"Subject": "hi", "Body": "see attachment", "Sender": "a@b.com", "Url": "", "File": "invoice.exe"},
        {"Subject": "hi", "Body": "link http://x.test/files/setup.zip now", "Sender": "a@b.com", "Url": "", "File": ""},
    ])
    feats = extract_structural_features(df)
    ext_col = 2
    assert feats[0, ext_col] == 0.0 and feats[1, ext_col] == 1.0 and feats[2, ext_col] == 1.0
