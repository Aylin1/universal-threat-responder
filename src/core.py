import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from src.domain_interface import ThreatSignal, DecisionOutput, DomainAdapter

logger = logging.getLogger(__name__)


class SecurityTools:
    """Domain-agnostic action execution."""

    def __init__(self, audit_log_path: str = 'logs/audit.log'):
        self.audit_log_path = audit_log_path
        Path(audit_log_path).parent.mkdir(exist_ok=True)
        self.stats = {
            'blocked': 0,
            'flagged': 0,
            'allowed': 0,
            'retrains_triggered': 0,
            'errors': 0
        }

    def block(self, signal_id: str, reason: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "BLOCK", reason, audit_id)
        self.stats['blocked'] += 1
        return audit_id

    def flag(self,
             signal_id: str,
             reason: str,
             priority: str = "medium") -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "FLAG", f"{reason} (Priority: {priority})",
                        audit_id)
        self.stats['flagged'] += 1
        return audit_id

    def allow(self, signal_id: str, confidence: float) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "ALLOW", f"confidence={confidence:.2%}",
                        audit_id)
        self.stats['allowed'] += 1
        return audit_id

    def trigger_retrain(self,
                        signal_id: str,
                        reason: str,
                        urgency: str = "normal") -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "RETRAIN", f"{reason} (Urgency: {urgency})",
                        audit_id)
        self.stats['retrains_triggered'] += 1
        return audit_id

    def _log_audit(self, signal_id: str, action: str, details: str,
                   audit_id: str):
        entry = {
            'audit_id': audit_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'signal_id': signal_id,
            'action': action,
            'details': details
        }
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return audit_id


# State schema for the LangGraph state machine
class AgentState(TypedDict):
    signal: ThreatSignal
    adapter: DomainAdapter
    formatted_prompt: str
    raw_response: str
    action: str
    reasoning: str
    audit_id: str
    latency_ms: float
    error: Optional[str]


class ThreatResponderGraph:
    """Stateful LangGraph agent for threat evaluation and action routing."""

    def __init__(self,
                 llm: BaseChatModel,
                 tools: Optional[SecurityTools] = None):
        self.llm = llm
        self.tools = tools or SecurityTools()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("build_prompt", self._node_build_prompt)
        builder.add_node("evaluate_threat", self._node_evaluate_threat)
        builder.add_node("parse_action", self._node_parse_action)
        builder.add_node("execute_action", self._node_execute_action)
        builder.add_node("fallback_handler", self._node_fallback_handler)

        builder.set_entry_point("build_prompt")
        builder.add_edge("build_prompt", "evaluate_threat")

        builder.add_conditional_edges(
            "evaluate_threat", lambda state: "fallback_handler"
            if state.get("error") else "parse_action", {
                "fallback_handler": "fallback_handler",
                "parse_action": "parse_action"
            })

        builder.add_edge("parse_action", "execute_action")
        builder.add_edge("execute_action", END)
        builder.add_edge("fallback_handler", END)

        return builder.compile()

    def _node_build_prompt(self, state: AgentState) -> Dict[str, Any]:
        adapter = state["adapter"]
        signal = state["signal"]
        template = adapter.get_prompt_template()
        prompt_text = template(signal,
                               adapter) if callable(template) else template
        return {"formatted_prompt": prompt_text}

    def _node_evaluate_threat(self, state: AgentState) -> Dict[str, Any]:
        from time import perf_counter
        start = perf_counter()
        try:
            response = self.llm.invoke(state["formatted_prompt"])
            response_text = getattr(response, "content", str(response))
            latency = (perf_counter() - start) * 1000
            return {
                "raw_response": response_text,
                "latency_ms": latency,
                "error": None
            }
        except Exception as e:
            return {"error": str(e), "latency_ms": 0}

    def _node_parse_action(self, state: AgentState) -> Dict[str, Any]:
        resp = state["raw_response"].lower()
        if 'block' in resp:
            action = 'BLOCK'
        elif 'allow' in resp:
            action = 'ALLOW'
        elif 'retrain' in resp:
            action = 'RETRAIN'
        else:
            action = 'FLAG'
        return {"action": action, "reasoning": state["raw_response"]}

    def _node_execute_action(self, state: AgentState) -> Dict[str, Any]:
        sig = state["signal"]
        act = state["action"]
        if act == 'BLOCK':
            audit_id = self.tools.block(sig.signal_id,
                                        f"HIGH RISK ({sig.risk_score:.2%})")
        elif act == 'ALLOW':
            audit_id = self.tools.allow(sig.signal_id, 1.0 - sig.risk_score)
        elif act == 'RETRAIN':
            audit_id = self.tools.trigger_retrain(sig.signal_id,
                                                  "Performance degradation")
        else:
            audit_id = self.tools.flag(
                sig.signal_id, f"Moderate risk ({sig.risk_score:.2%})")
        return {"audit_id": audit_id}

    def _node_fallback_handler(self, state: AgentState) -> Dict[str, Any]:
        sig = state["signal"]
        act = 'BLOCK' if sig.risk_score > 0.90 else (
            'FLAG' if sig.risk_score > 0.70 else 'ALLOW')
        audit_id = self.tools.flag(
            sig.signal_id, f"Fallback due to error: {state.get('error')}")
        return {
            "action": act,
            "reasoning":
            f"Fallback executed due to error: {state.get('error')}",
            "audit_id": audit_id
        }

    def respond(self, signal: ThreatSignal,
                adapter: DomainAdapter) -> DecisionOutput:
        initial_state = {
            "signal": signal,
            "adapter": adapter,
            "formatted_prompt": "",
            "raw_response": "",
            "action": "",
            "reasoning": "",
            "audit_id": "",
            "latency_ms": 0.0,
            "error": None
        }
        final_state = self.graph.invoke(initial_state)
        return DecisionOutput(action=final_state["action"],
                              reasoning=final_state["reasoning"],
                              audit_id=final_state["audit_id"],
                              latency_ms=final_state["latency_ms"])
