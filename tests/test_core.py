import json
from unittest.mock import MagicMock

import pytest

from config import cfg
from src.core import SecurityTools, ThreatResponderGraph, route_threat_signal
from src.domain_interface import ThreatSignal


class MockAdapter:
    def get_prompt_template(self):
        return lambda signal, adapter: "Analyze the threat"


def make_signal(
    risk_score: float,
    *,
    predicted_class: str = "QUARANTINE",
    max_risk_score: float = 0.0,
) -> ThreatSignal:
    return ThreatSignal(
        signal_id="test-signal",
        risk_score=risk_score,
        confidence=0.9,
        raw_payload="Sample threat",
        context={
            "ml_predicted_class": predicted_class,
            "max_risk_score": max_risk_score,
        },
    )


def test_router_uses_configured_high_bound(monkeypatch):
    monkeypatch.setattr(cfg.conformal, "high_bound", 0.95)

    assert route_threat_signal({"signal": make_signal(0.95)}) == "fast_path_bypass"
    assert route_threat_signal({"signal": make_signal(0.949)}) == "escalate_to_llm"


def test_fast_path_uses_ml_class_and_skips_llm(tmp_path):
    llm = MagicMock()
    tools = SecurityTools(str(tmp_path / "audit.jsonl"))
    graph = ThreatResponderGraph(llm=llm, tools=tools)

    decision = graph.respond(
        make_signal(0.96, predicted_class="PHISHING"),
        MockAdapter(),
    )

    assert decision.action == "PHISHING"
    assert decision.metadata["evaluation_mode"] == "FAST_PATH_BYPASS"
    llm.invoke.assert_not_called()

    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["action"] == "BLOCK"
    assert audit["signal_id"] == "test-signal"


def test_uncertain_signal_escalates_and_logs_valid_decision(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value.content = (
        '{"action":"VALID","confidence":0.9,"reasoning":"Expected message."}'
    )
    tools = SecurityTools(str(tmp_path / "audit.jsonl"))
    graph = ThreatResponderGraph(llm=llm, tools=tools)

    decision = graph.respond(make_signal(0.7), MockAdapter())

    assert decision.action == "VALID"
    assert decision.metadata["evaluation_mode"] == "ESCALATION_RAG_LLM"
    llm.invoke.assert_called_once_with("Analyze the threat")
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["action"] == "ALLOW"


def test_llm_exception_fails_closed_and_is_audited(tmp_path):
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("model unavailable")
    tools = SecurityTools(str(tmp_path / "audit.jsonl"))
    graph = ThreatResponderGraph(llm=llm, tools=tools)

    decision = graph.respond(make_signal(0.5), MockAdapter())

    assert decision.action == "QUARANTINE"
    assert decision.confidence == 0.0
    assert decision.metadata["evaluation_mode"] == "fallback_handler"
    assert "model unavailable" in decision.metadata["error"]
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8"))
    assert audit["action"] == "QUARANTINE"


@pytest.mark.parametrize(
    "response",
    ["", "not json", '{"action":"unknown","confidence":0.8}'],
)
def test_unparseable_llm_response_defaults_to_quarantine(tmp_path, response):
    llm = MagicMock()
    llm.invoke.return_value.content = response
    graph = ThreatResponderGraph(
        llm=llm,
        tools=SecurityTools(str(tmp_path / "audit.jsonl")),
        use_fast_path=False,
    )

    decision = graph.respond(make_signal(0.5), MockAdapter())

    assert decision.action == "QUARANTINE"


def test_invalid_action_is_not_mapped_to_valid(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value.content = '{"action":"INVALID","confidence":0.8}'
    graph = ThreatResponderGraph(
        llm=llm,
        tools=SecurityTools(str(tmp_path / "audit.jsonl")),
        use_fast_path=False,
    )

    decision = graph.respond(make_signal(0.5), MockAdapter())

    assert decision.action == "QUARANTINE"


def test_non_numeric_confidence_keeps_parsed_action(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value.content = (
        '{"action":"VALID","confidence":"high","reasoning":"Not phishing or spam."}'
    )
    graph = ThreatResponderGraph(
        llm=llm,
        tools=SecurityTools(str(tmp_path / "audit.jsonl")),
        use_fast_path=False,
    )

    decision = graph.respond(make_signal(0.5), MockAdapter())

    assert decision.action == "VALID"
    assert decision.confidence == 0.5


def test_low_confidence_llm_decision_is_quarantined(tmp_path):
    llm = MagicMock()
    llm.invoke.return_value.content = '{"action":"VALID","confidence":0.3,"reasoning":"Unsure."}'
    graph = ThreatResponderGraph(
        llm=llm,
        tools=SecurityTools(str(tmp_path / "audit.jsonl")),
        confidence_threshold=0.5,
        use_fast_path=False,
    )

    decision = graph.respond(make_signal(0.5), MockAdapter())

    assert decision.action == "QUARANTINE"
    assert decision.metadata["low_confidence_guard_applied"] is True
    assert decision.metadata["guard_applied"] == "low_confidence"
