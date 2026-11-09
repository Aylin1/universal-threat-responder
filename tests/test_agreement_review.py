import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evals.analyze_results import summarize, to_markdown
from evals.sweep_guard import replay_agreement_review
from src.core import SecurityTools, ThreatResponderGraph
from src.domain_interface import DomainAdapter, ThreatSignal
from src.guard import GuardConfig, evaluate_guard


class EmailAdapter(DomainAdapter):
    domain_name = "email"
    domain_description = "test"
    allowed_actions = ("VALID", "SPAM", "PHISHING", "QUARANTINE")
    benign_action = "VALID"
    ioc_block_action = "PHISHING"
    action_effects = {"VALID": "allow", "SPAM": "quarantine", "PHISHING": "block",
                      "QUARANTINE": "quarantine"}

    def extract_features(self, raw):
        return raw

    def build_context(self, raw, metadata=None):
        return {}

    def validate_input(self, raw):
        return True

    def get_prompt_template(self):
        return lambda signal, adapter: "classify"


@pytest.mark.parametrize(
    "ml,cset,llm,expected",
    [
        ("PHISHING", ["PHISHING"], "PHISHING", "PHISHING"),
        ("SPAM", ["SPAM"], "SPAM", "SPAM"),
        ("VALID", ["VALID"], "VALID", "VALID"),
        ("VALID", ["VALID"], "PHISHING", "QUARANTINE"),
        ("PHISHING", ["PHISHING"], "VALID", "QUARANTINE"),
        ("SPAM", ["SPAM"], "PHISHING", "QUARANTINE"),
        ("VALID", ["VALID", "SPAM"], "VALID", "QUARANTINE"),
        ("SPAM", [], "SPAM", "QUARANTINE"),
        ("SPAM", ["PHISHING"], "SPAM", "QUARANTINE"),
    ],
)
def test_agreement_review_decisions(tmp_path, ml, cset, llm, expected):
    model = MagicMock()
    model.invoke.return_value.content = json.dumps({"action": llm, "confidence": 0.9})
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    policies = [{"sop_id": "SOP-103", "sim": 0.9}]
    signal = ThreatSignal("S", 0.9, 0.9, {
        "ml_predicted_class": ml, "conformal_set": cset,
        "conformal_singleton": len(cset) == 1, "review_policies": policies,
    })
    decision = graph.respond(signal, EmailAdapter())
    assert decision.action == expected
    assert decision.metadata["review_policies"] == policies
    assert decision.metadata["self_reported_confidence"] == 0.9
    assert decision.metadata["llm_confidence_source"] == "model_reported"
    model.invoke.assert_called_once()
    audit = json.loads((tmp_path / "audit.jsonl").read_text())
    assert audit["action"] == ("ALLOW" if expected == "VALID" else
                               "BLOCK" if expected == "PHISHING" else "QUARANTINE")


def test_agreed_valid_remains_quarantined_by_neighbor_guard(tmp_path):
    class GuardedAdapter(EmailAdapter):
        def apply_policy_guard(self, signal, action):
            verdict = evaluate_guard(
                action, ml_class="VALID", ml_prob=0.9,
                neighbors=[{"label": "PHISHING", "sim": 0.9}] * 2,
                cfg=GuardConfig(),
            )
            return verdict.action, verdict.applied, verdict.reasons

    model = MagicMock()
    model.invoke.return_value.content = '{"action":"VALID","confidence":0.9}'
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    signal = ThreatSignal("S", 0.9, 0.9, {
        "ml_predicted_class": "VALID", "conformal_set": ["VALID"], "conformal_singleton": True,
    })
    result = graph.respond(signal, GuardedAdapter())
    assert result.action == "QUARANTINE"
    assert result.metadata["policy_guard_applied"] is True


def test_ioc_does_not_bypass_agreement(tmp_path):
    model = MagicMock()
    model.invoke.return_value.content = '{"action":"SPAM","confidence":0.9}'
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    signal = ThreatSignal("S", 0.9, 0.9, {
        "ml_predicted_class": "SPAM", "conformal_singleton": True, "conformal_set": ["SPAM"],
        "ioc_hits": 1, "ioc_matches": [{"match_type": "url"}],
    })
    result = graph.respond(signal, EmailAdapter())
    assert result.action == "QUARANTINE"
    model.invoke.assert_called_once()


def test_llm_failure_is_quarantined_with_policy_context(tmp_path):
    model = MagicMock()
    model.invoke.side_effect = RuntimeError("model unavailable")
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    policies = [{"sop_id": "SOP-103", "sim": 0.9}]
    signal = ThreatSignal("S", 0.9, 0.9, {"review_policies": policies})
    result = graph.respond(signal, EmailAdapter())
    assert result.action == "QUARANTINE"
    assert result.metadata["error"] == "model unavailable"
    assert result.metadata["review_policies"] == policies


def test_agreed_valid_below_llm_confidence_threshold_is_quarantined(tmp_path):
    model = MagicMock()
    model.invoke.return_value.content = '{"action":"VALID","confidence":0.1}'
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    signal = ThreatSignal("S", 0.9, 0.9, {
        "ml_predicted_class": "VALID", "conformal_set": ["VALID"], "conformal_singleton": True,
    })
    result = graph.respond(signal, EmailAdapter())
    assert result.action == "QUARANTINE"
    assert result.metadata["abstain_reason"]


@pytest.mark.parametrize("confidence", [None, "high", "NaN"])
def test_missing_or_invalid_llm_confidence_is_labeled_default(tmp_path, confidence):
    model = MagicMock()
    model.invoke.return_value.content = json.dumps({"action": "VALID", "confidence": confidence})
    graph = ThreatResponderGraph(model, tools=SecurityTools(str(tmp_path / "audit.jsonl")),
                                 use_fast_path=False, require_agreement_review=True)
    signal = ThreatSignal("S", 0.9, 0.9, {
        "ml_predicted_class": "VALID", "conformal_set": ["VALID"], "conformal_singleton": True,
    })
    result = graph.respond(signal, EmailAdapter())
    assert result.action == "VALID"
    assert result.metadata["self_reported_confidence"] == 0.5
    assert result.metadata["llm_confidence_source"] == "contains_default"


def test_replay_agreed_valid_and_disagreement():
    df = pd.DataFrame([
        {"agent_action": "VALID", "llm_raw_action": "VALID", "ml_predicted_class": "VALID",
         "conformal_set": "VALID", "conformal_singleton": True, "ioc_hits": 0},
        {"agent_action": "VALID", "llm_raw_action": "VALID", "ml_predicted_class": "SPAM",
         "conformal_set": "SPAM", "conformal_singleton": True, "ioc_hits": 0},
        {"agent_action": "QUARANTINE", "llm_raw_action": "VALID", "ml_predicted_class": "VALID",
         "conformal_set": "VALID", "conformal_singleton": True, "ioc_hits": 0},
    ])
    assert replay_agreement_review(df).tolist() == ["VALID", "QUARANTINE", "QUARANTINE"]


def test_agreement_cannot_enable_fast_path():
    with pytest.raises(ValueError, match="fast path"):
        ThreatResponderGraph(MagicMock(), use_fast_path=True, require_agreement_review=True)


def test_replay_rejects_missing_predictions():
    df = pd.DataFrame([{"agent_action": "VALID", "llm_raw_action": None, "ml_predicted_class": "VALID",
                        "conformal_set": "VALID", "conformal_singleton": True, "ioc_hits": 0}])
    with pytest.raises(ValueError, match="every row"):
        replay_agreement_review(df)


def test_partial_ml_predictions_do_not_produce_misleading_baseline():
    df = pd.DataFrame([
        {"true_label": "VALID", "agent_action": "VALID", "route_taken": "FAST_PATH_BYPASS",
         "ml_predicted_class": "VALID", "ml_top_prob": 0.9},
        {"true_label": "SPAM", "agent_action": "SPAM", "route_taken": "FAST_PATH_BYPASS",
         "ml_predicted_class": None, "ml_top_prob": None},
    ])
    summary = summarize(df)
    assert not any(m["system"].startswith("ML only") for m in summary["systems"])
    assert "ML baseline omitted" in to_markdown(summary)


def test_conformal_baseline_uses_singleton_not_argmax():
    df = pd.DataFrame([{
        "true_label": "SPAM", "agent_action": "SPAM", "route_taken": "FAST_PATH_BYPASS",
        "ml_predicted_class": "VALID", "ml_top_prob": 0.5,
        "conformal_set": "SPAM", "conformal_singleton": True,
    }])
    summary = summarize(df)
    baseline = next(m for m in summary["systems"] if m["system"].startswith("ML + conformal"))
    assert baseline["resolved_accuracy"] == 1.0
