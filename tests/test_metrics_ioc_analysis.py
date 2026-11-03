import json

import numpy as np
import pandas as pd

from src.ioc import IOCIndex, normalize_url
from src.metrics import (bootstrap_ci, brier_score, distribution_entropy, expected_calibration_error,
                         l2_to_cosine_similarity, top2_margin)
from evals.analyze_results import summarize, to_markdown


# ----------------------------------------------------------------------------- metrics
def test_distribution_entropy_known_values():
    assert abs(distribution_entropy([1 / 3] * 3) - np.log2(3)) < 1e-9
    assert distribution_entropy([1.0, 0.0, 0.0]) < 1e-6
    # binary entropy of top prob would call [0.5,0.25,0.25] and [0.5,0.5,0.0] identical; the distribution doesn't
    assert distribution_entropy([0.5, 0.25, 0.25]) > distribution_entropy([0.5, 0.5, 0.0])


def test_margin():
    assert abs(top2_margin([0.6, 0.3, 0.1]) - 0.3) < 1e-9


def test_ece_perfectly_calibrated_and_overconfident():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 20000)
    correct = rng.uniform(size=20000) < conf
    ece, _ = expected_calibration_error(conf, correct)
    assert ece < 0.02
    ece_over, _ = expected_calibration_error(np.full(1000, 0.95), np.r_[np.ones(500), np.zeros(500)])
    assert abs(ece_over - 0.45) < 1e-9


def test_brier_and_bootstrap():
    assert abs(brier_score([1.0, 0.0], [1, 0])) < 1e-12
    m, lo, hi = bootstrap_ci([1, 0] * 500, n_boot=500)
    assert lo < m < hi and abs(m - 0.5) < 1e-9


def test_l2_cosine_mapping():
    assert abs(l2_to_cosine_similarity(0.0) - 1.0) < 1e-12
    assert abs(l2_to_cosine_similarity(2.0) - 0.0) < 1e-12
    assert abs(l2_to_cosine_similarity(0.84) - 0.58) < 1e-9


# ----------------------------------------------------------------------------- IOC
def _idx():
    return IOCIndex([
        {"indicator_id": "OPENPHISH-1", "url": "http://Evil.example.com/login/", "source": "openphish_live"},
        {"indicator_id": "URLHAUS-9", "url": "https://bad.test/payload.exe", "source": "urlhaus_live"},
    ])


def test_ioc_exact_match_is_scheme_case_and_slash_insensitive():
    hits = _idx().match("please visit https://evil.example.com/login now.")
    assert len(hits) == 1 and hits[0]["match_type"] == "url" and hits[0]["indicator_id"] == "OPENPHISH-1"


def test_ioc_host_only_match_is_soft():
    hits = _idx().match("see http://evil.example.com/some/other/page")
    assert len(hits) == 1 and hits[0]["match_type"] == "host"


def test_ioc_no_boilerplate_false_positive():
    # The old embedding-based feed docs matched any 'account/login' wording. Exact matching cannot.
    assert _idx().match("Your account login details changed. Click here to review your credentials.") == []


def test_normalize_url():
    assert normalize_url("HTTPS://A.com/x/#frag") == normalize_url("http://a.com/x")


def test_ioc_roundtrip_from_json(tmp_path):
    p = tmp_path / "f.json"
    p.write_text(json.dumps({"fetched_at": "t", "records": [{"url": "http://a.test/x", "indicator_id": "I1"}]}))
    assert len(IOCIndex.from_json(str(p))) == 1


# ----------------------------------------------------------------------------- analysis
def _frame():
    rows = []
    # 6 fast-path rows (5 correct), 4 LLM-path rows (1 abstain, 1 guard veto of a bad VALID, 1 correct, 1 wrong)
    for i in range(6):
        t = "SPAM" if i % 2 else "VALID"
        p = t if i < 5 else ("VALID" if t == "SPAM" else "SPAM")
        rows.append(dict(true_label=t, agent_action=p, llm_raw_action=None, route_taken="FAST_PATH_BYPASS",
                         agent_confidence=0.97, ml_predicted_class=p, ml_top_prob=0.97,
                         conformal_singleton=True, guard_applied=False, abstained=False))
    rows.append(dict(true_label="PHISHING", agent_action="QUARANTINE", llm_raw_action="VALID",
                     route_taken="ESCALATION_RAG_LLM", agent_confidence=0.9, ml_predicted_class="PHISHING",
                     ml_top_prob=0.7, conformal_singleton=False, guard_applied=True, abstained=False))
    rows.append(dict(true_label="SPAM", agent_action="QUARANTINE", llm_raw_action="SPAM",
                     route_taken="ESCALATION_RAG_LLM", agent_confidence=0.5, ml_predicted_class="SPAM",
                     ml_top_prob=0.6, conformal_singleton=False, guard_applied=False, abstained=True))
    rows.append(dict(true_label="VALID", agent_action="VALID", llm_raw_action="VALID",
                     route_taken="ESCALATION_RAG_LLM", agent_confidence=0.9, ml_predicted_class="SPAM",
                     ml_top_prob=0.55, conformal_singleton=False, guard_applied=False, abstained=False))
    rows.append(dict(true_label="SPAM", agent_action="PHISHING", llm_raw_action="PHISHING",
                     route_taken="ESCALATION_RAG_LLM", agent_confidence=0.8, ml_predicted_class="SPAM",
                     ml_top_prob=0.52, conformal_singleton=False, guard_applied=False, abstained=False))
    return pd.DataFrame(rows)


def test_summary_counts_abstention_as_error_end_to_end():
    s = summarize(_frame())
    full = [m for m in s["systems"] if m["system"].startswith("FULL:")][0]
    # decided = 10 - 2 quarantines = 8; correct = 5 fast + 1 valid = 6
    assert abs(full["coverage"] - 0.8) < 1e-9
    assert abs(full["resolved_accuracy"] - 6 / 8) < 1e-9
    assert abs(full["end_to_end_accuracy"] - 6 / 10) < 1e-9
    assert full["end_to_end_accuracy"] < full["resolved_accuracy"]  # the old headline hid this gap


def test_summary_has_ml_baseline_and_counterfactual_guard():
    s = summarize(_frame())
    names = [m["system"] for m in s["systems"]]
    assert any(n.startswith("ML only") for n in names)
    assert any("without the policy guard" in n for n in names)
    ng = [m for m in s["systems"] if "without the policy guard" in m["system"]][0]
    full = [m for m in s["systems"] if m["system"].startswith("FULL:")][0]
    # 6 spam/phishing rows; the fast path leaks 1; without the guard the PHISHING row would leak too
    assert abs(full["leak_rate"] - 1 / 6) < 1e-9
    assert abs(ng["leak_rate"] - 2 / 6) < 1e-9


def test_guard_impact_and_markdown_render():
    s = summarize(_frame())
    assert s["guard"]["vetoed"] == 1 and s["guard"]["vetoed_and_actually_bad"] == 1
    md = to_markdown(s)
    assert "End-to-end acc." in md and "Confusion matrix" in md
