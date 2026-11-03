import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from src.core import SecurityTools, ThreatResponderGraph, route_threat_signal  # noqa: E402
from src.domain_interface import DomainAdapter, ThreatSignal  # noqa: E402


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, prompt):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return type("R", (), {"content": r})()


class SpamAdapter(DomainAdapter):
    """Minimal spam-domain adapter (no Milvus) reusing the real guard."""
    domain_name = "spam_phishing"
    domain_description = "t"

    def __init__(self):
        from src.guard import GuardConfig, evaluate_guard
        self._eg, self._gc = evaluate_guard, GuardConfig()

    allowed_actions = ("VALID", "SPAM", "PHISHING", "QUARANTINE")
    benign_action = "VALID"
    ioc_block_action = "PHISHING"
    action_effects = {"PHISHING": "block", "SPAM": "quarantine", "VALID": "allow", "QUARANTINE": "quarantine"}

    def extract_features(self, raw): return raw
    def build_context(self, raw, metadata=None): return {}
    def validate_input(self, raw): return True
    def get_prompt_template(self): return lambda s, a: "prompt"

    def apply_policy_guard(self, signal, action):
        c = signal.context
        v = self._eg(action, ml_class=c.get("ml_predicted_class", ""),
                     ml_prob=c.get("ml_probs", {}).get(c.get("ml_predicted_class", ""), 0.0),
                     neighbors=c.get("neighbors", []), ioc_hits=c.get("ioc_hits", 0), cfg=self._gc)
        return v.action, v.applied, v.reasons


def _sig(**ctx):
    base = {"ml_predicted_class": "VALID", "ml_probs": {"VALID": 0.9, "SPAM": 0.05, "PHISHING": 0.05}}
    base.update(ctx)
    return ThreatSignal(signal_id="S1", risk_score=0.9, confidence=0.9, context=base)


def _run(llm_replies, sig, tmp_path, **kw):
    g = ThreatResponderGraph(FakeLLM(llm_replies), tools=SecurityTools(str(tmp_path / "a.log")), **kw)
    return g.respond(sig, SpamAdapter())


def test_routing_precedence():
    st = lambda s: {"signal": s}
    assert route_threat_signal(st(_sig(conformal_singleton=True))) == "fast_path_bypass"
    assert route_threat_signal(st(_sig(conformal_singleton=False))) == "escalate_to_llm"
    assert route_threat_signal(st(_sig(conformal_singleton=True)), use_fast_path=False) == "escalate_to_llm"
    ioc = [{"match_type": "url", "indicator_id": "X"}]
    assert route_threat_signal(st(_sig(conformal_singleton=True, ioc_matches=ioc))) == "ioc_block"
    assert route_threat_signal(st(_sig(ioc_matches=ioc)), use_fast_path=False) == "ioc_block"
    host_only = [{"match_type": "host"}]
    assert route_threat_signal(st(_sig(conformal_singleton=False, ioc_matches=host_only))) == "escalate_to_llm"


def test_ioc_hit_blocks_without_llm(tmp_path):
    llm = FakeLLM(['{"action":"VALID"}'])
    g = ThreatResponderGraph(llm, tools=SecurityTools(str(tmp_path / "a.log")))
    out = g.respond(_sig(ioc_matches=[{"match_type": "url", "indicator_id": "X"}]), SpamAdapter())
    assert out.action == "PHISHING" and llm.calls == 0 and out.metadata["evaluation_mode"] == "IOC_BLOCK"


def test_fast_path_skips_llm(tmp_path):
    llm = FakeLLM(["x"])
    g = ThreatResponderGraph(llm, tools=SecurityTools(str(tmp_path / "a.log")))
    out = g.respond(_sig(conformal_singleton=True, conformal_set=["VALID"]), SpamAdapter())
    assert out.action == "VALID" and llm.calls == 0


def test_llm_path_and_guard_veto_by_ml(tmp_path):
    sig = _sig(conformal_singleton=False, ml_predicted_class="PHISHING",
               ml_probs={"VALID": 0.1, "SPAM": 0.1, "PHISHING": 0.8})
    out = _run(['{"action":"VALID","confidence":0.9,"reasoning":"fine"}'], sig, tmp_path)
    assert out.action == "QUARANTINE" and out.metadata["policy_guard_applied"] is True
    assert out.metadata["llm_raw_action"] == "VALID"


def test_guard_does_not_fire_on_weak_matches_the_old_failure_mode(tmp_path):
    weak = [{"label": "PHISHING", "sim": 0.58}] * 8
    out = _run(['{"action":"VALID","confidence":0.9,"reasoning":"routine"}'],
               _sig(conformal_singleton=False, neighbors=weak), tmp_path)
    assert out.action == "VALID" and out.metadata["policy_guard_applied"] is False


def test_unparseable_and_negated_text_abstain(tmp_path):
    out = _run(["This is not phishing, it looks fine."], _sig(conformal_singleton=False), tmp_path)
    assert out.action == "QUARANTINE" and out.metadata["parse_failed"] is True


def test_low_confidence_abstains(tmp_path):
    out = _run(['{"action":"SPAM","confidence":0.2,"reasoning":"meh"}'],
               _sig(conformal_singleton=False), tmp_path, confidence_threshold=0.5)
    assert out.action == "QUARANTINE" and "below threshold" in out.reasoning


def test_self_consistency_majority_and_no_majority(tmp_path):
    spam = '{"action":"SPAM","confidence":0.9,"reasoning":"a"}'
    valid = '{"action":"VALID","confidence":0.9,"reasoning":"b"}'
    phish = '{"action":"PHISHING","confidence":0.9,"reasoning":"c"}'
    out = _run([spam, spam, valid], _sig(conformal_singleton=False), tmp_path, uncertainty_samples=3)
    assert out.action == "SPAM" and abs(out.metadata["vote_agreement"] - 2 / 3) < 1e-9
    out2 = _run([spam, valid, phish], _sig(conformal_singleton=False), tmp_path, uncertainty_samples=3)
    assert out2.action == "QUARANTINE" and "no majority" in out2.reasoning


def test_llm_exception_falls_back_to_quarantine(tmp_path):
    class Boom:
        def invoke(self, p): raise RuntimeError("ollama down")
    g = ThreatResponderGraph(Boom(), tools=SecurityTools(str(tmp_path / "a.log")))
    assert g.respond(_sig(conformal_singleton=False), SpamAdapter()).action == "QUARANTINE"
