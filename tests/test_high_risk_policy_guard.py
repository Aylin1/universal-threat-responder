from unittest.mock import MagicMock

from src.core import SecurityTools, ThreatResponderGraph
from src.domain_interface import ThreatSignal


class MockAdapter:
    def get_prompt_template(self):
        return lambda signal, adapter: "Analyze the threat"


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