from unittest.mock import MagicMock

import pytest

from src.core import SecurityTools, ThreatResponderGraph
from src.domain_interface import ThreatSignal


class MockAdapter:
    def get_prompt_template(self):
        return lambda signal, adapter: "Analyze the threat"


def run_valid_verdict(tmp_path, context):
    llm = MagicMock()
    llm.invoke.return_value.content = (
        '{"action": "VALID", "confidence": 0.9, "reasoning": "Legitimate correspondence."}'
    )
    graph = ThreatResponderGraph(
        llm=llm,
        tools=SecurityTools(str(tmp_path / "audit.log")),
        use_fast_path=False,
    )
    signal = ThreatSignal(
        signal_id="guard-evidence",
        risk_score=0.7,
        confidence=0.7,
        raw_payload="Quarterly planning notes",
        context={"raw_input": "Quarterly planning notes", "max_risk_score": 0.95, **context},
    )
    return graph.respond(signal, MockAdapter())


@pytest.mark.parametrize(
    "context",
    [
        {"ioc_match": True, "high_risk_similarity": 0.60, "ml_predicted_class": "VALID"},
        {"ioc_match": False, "high_risk_similarity": 0.85, "ml_predicted_class": "VALID"},
        {"ioc_match": False, "high_risk_similarity": 0.60, "ml_predicted_class": "PHISHING"},
        {"ioc_match": False, "high_risk_similarity": 0.60, "ml_predicted_class": "SPAM"},
    ],
)
def test_guard_quarantines_valid_when_evidence_corroborates(tmp_path, context):
    decision = run_valid_verdict(tmp_path, context)

    assert decision.action == "QUARANTINE"
    assert decision.metadata["policy_guard_applied"] is True
    assert decision.metadata["guard_applied"] == "policy"


def test_guard_allows_valid_when_high_risk_match_is_only_topical(tmp_path):
    decision = run_valid_verdict(
        tmp_path,
        {"ioc_match": False, "high_risk_similarity": 0.60, "ml_predicted_class": "VALID"},
    )

    assert decision.action == "VALID"
    assert decision.metadata["policy_guard_applied"] is False
    assert decision.metadata["guard_applied"] == "none"


def test_guard_without_evidence_fields_fails_closed(tmp_path):
    decision = run_valid_verdict(tmp_path, {})

    assert decision.action == "QUARANTINE"
    assert decision.metadata["guard_applied"] == "policy"


def test_high_retrieved_risk_cannot_be_marked_valid(tmp_path):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = (
        '{"action": "VALID", "confidence": 0.98, '
        '"reasoning": "Legitimate business correspondence."}'
    )
    graph = ThreatResponderGraph(
        llm=mock_llm,
        tools=SecurityTools(str(tmp_path / "audit.log")),
        use_fast_path=False,
    )
    signal = ThreatSignal(
        signal_id="high-risk-valid-conflict",
        risk_score=0.96,
        confidence=0.98,
        raw_payload="Consulting project opportunity",
        context={
            "raw_input": "Consulting project opportunity",
            "max_risk_score": 0.96,
            "high_risk_hits": 5,
        },
    )

    decision = graph.respond(signal, MockAdapter())

    assert decision.action == "QUARANTINE"
    assert decision.metadata["policy_guard_applied"] is True
    assert "high-risk bound" in decision.reasoning