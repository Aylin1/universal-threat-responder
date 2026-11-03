import json

import numpy as np
import pandas as pd

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter, extract_structural_features

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


def test_suspicious_extension_is_not_a_substring_match():
    df = pd.DataFrame([
        {"Subject": "hi", "Body": "visit mail.executivegroup.com today", "Sender": "a@b.com", "Url": "", "File": ""},
        {"Subject": "hi", "Body": "see attachment", "Sender": "a@b.com", "Url": "", "File": "invoice.exe"},
        {"Subject": "hi", "Body": "link http://x.test/files/setup.zip now", "Sender": "a@b.com", "Url": "", "File": ""},
    ])
    feats = extract_structural_features(df)
    ext_col = 2
    assert feats[0, ext_col] == 0.0 and feats[1, ext_col] == 1.0 and feats[2, ext_col] == 1.0
