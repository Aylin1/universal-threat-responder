import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TypedDict
import re
from time import perf_counter, process_time

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from src.domain_interface import ThreatSignal, DecisionOutput, DomainAdapter
from adapters.rag_spam_adapter import compute_shannon_entropy
from config import cfg

logger = logging.getLogger(__name__)


class SecurityTools:
    def __init__(self, audit_log_path: str = 'logs/audit.log'):
        self.audit_log_path = audit_log_path
        Path(audit_log_path).parent.mkdir(parents=True, exist_ok=True)
        self.stats = {'blocked': 0, 'quarantined': 0, 'allowed': 0, 'errors': 0}

    def block(self, signal_id: str, reason: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "BLOCK", reason, audit_id)
        self.stats['blocked'] += 1
        return audit_id

    def quarantine(self, signal_id: str, reason: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "QUARANTINE", reason, audit_id)
        self.stats['quarantined'] += 1
        return audit_id

    def allow(self, signal_id: str, confidence: float) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "ALLOW", f"confidence={confidence:.2%}", audit_id)
        self.stats['allowed'] += 1
        return audit_id

    def flag(self, signal_id: str, reason: str, priority: str = "medium") -> str:
        return self.quarantine(signal_id, f"{reason} (Priority: {priority})")

    def _log_audit(self, signal_id: str, action: str, details: str, audit_id: str):
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
    confidence: float
    uncertainty_metadata: Dict[str, Any]


def route_threat_signal(
    state: AgentState,
    high_accuracy_bound: float = 0.95,
) -> str:
    sig = state["signal"]
    max_prob = getattr(sig, "risk_score", 0.50)
    # Pull bounds dynamically from config to prevent over-triggering fast-path
    hb = getattr(cfg.conformal, "high_bound", 0.95)
    use_fp = getattr(cfg.conformal, "use_fast_path", True)

    # Only trigger fast-path if explicitly enabled and outside the ambiguity window
    if use_fp and max_prob >= hb:
        return "fast_path_bypass"
    return "escalate_to_llm"


class ThreatResponderGraph:
    def __init__(
        self, 
        llm: BaseChatModel, 
        tools: Optional[SecurityTools] = None, 
        uncertainty_samples: int = 1, 
        confidence_threshold: float = 0.50,
        use_fast_path: bool = True
    ):
        self.llm = llm
        self.tools = tools or SecurityTools()
        self.uncertainty_samples = max(1, uncertainty_samples)
        self.confidence_threshold = confidence_threshold
        self.use_fast_path = use_fast_path
        self.graph = self._build_graph()

    def _get_device_name(self) -> str:
        try:
            import torch
            return "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except ImportError:
            return "CPU"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("fast_path_bypass", self._node_fast_path_bypass)
        builder.add_node("build_prompt", self._node_build_prompt)
        builder.add_node("evaluate_threat", self._node_evaluate_threat)
        builder.add_node("execute_action", self._node_execute_action)
        builder.add_node("fallback_handler", self._node_fallback_handler)

        if self.use_fast_path:
            builder.add_conditional_edges(
                START, 
                route_threat_signal, 
                {
                    "fast_path_bypass": "fast_path_bypass",
                    "escalate_to_llm": "build_prompt"
                }
            )
        else:
            builder.add_edge(START, "build_prompt")

        builder.add_edge("fast_path_bypass", "execute_action")
        builder.add_edge("build_prompt", "evaluate_threat")
        builder.add_conditional_edges(
            "evaluate_threat", 
            lambda state: "fallback_handler" if state.get("error") else "execute_action", 
            {
                "fallback_handler": "fallback_handler",
                "execute_action": "execute_action"
            }
        )
        builder.add_edge("execute_action", END)
        builder.add_edge("fallback_handler", END)
        return builder.compile()

    def _node_fast_path_bypass(self, state: AgentState) -> Dict[str, Any]:
        start_wall, start_cpu = perf_counter(), process_time()
        sig = state["signal"]
        confidence = getattr(sig, "risk_score", 0.50)
        entropy = compute_shannon_entropy(confidence)

        pred_class = sig.context.get("ml_predicted_class", "QUARANTINE")
        reasoning = f"Fast-path classified as {pred_class} via high calibrated probability ({confidence:.4f})."
        raw_resp = json.dumps({"action": pred_class, "confidence": confidence, "reasoning": reasoning})

        wall_latency = (perf_counter() - start_wall) * 1000
        cpu_latency = (process_time() - start_cpu) * 1000

        return {
            "action": pred_class,
            "raw_response": raw_resp,
            "reasoning": reasoning,
            "confidence": confidence,
            "latency_ms": wall_latency,
            "error": None,
            "uncertainty_metadata": {
                "evaluation_mode": "FAST_PATH_BYPASS",
                "risk_score": confidence,
                "shannon_entropy_bits": entropy,
                "cpu_latency_ms": round(cpu_latency, 2),
                "wall_latency_ms": round(wall_latency, 2),
                "hardware_device": self._get_device_name()
            }
        }

    def _node_build_prompt(self, state: AgentState) -> Dict[str, Any]:
        adapter = state["adapter"]
        signal = state["signal"]
        template = adapter.get_prompt_template()
        prompt_text = template(signal, adapter) if callable(template) else template
        return {"formatted_prompt": prompt_text}

    def _extract_action_from_json(self, raw_text: str) -> Dict[str, Any]:
        default_fail = {
            "action": "QUARANTINE", 
            "confidence": 0.5, 
            "reasoning": "Failed to parse LLM response; defaulting to QUARANTINE"
        }

        if not raw_text or not isinstance(raw_text, str):
            return default_fail

        cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                act_val = parsed.get("action") or parsed.get("decision") or parsed.get("classification") or ""
                action = str(act_val).upper().strip()

                if action not in ["VALID", "SPAM", "PHISHING", "QUARANTINE"]:
                    if "PHISH" in action:
                        action = "PHISHING"
                    elif "SPAM" in action:
                        action = "SPAM"
                    elif "VALID" in action or "ALLOW" in action:
                        action = "VALID"
                    else:
                        action = "QUARANTINE"

                return {
                    "action": action,
                    "confidence": float(parsed.get("confidence", 0.85)),
                    "reasoning": str(parsed.get("reasoning", "Parsed successfully via JSON extractor."))
                }
            except Exception:
                pass

        upper_text = raw_text.upper()
        if "PHISH" in upper_text or "BLOCK" in upper_text:
            return {"action": "PHISHING", "confidence": 0.75, "reasoning": "Parsed via heuristic text fallback."}
        elif "SPAM" in upper_text:
            return {"action": "SPAM", "confidence": 0.75, "reasoning": "Parsed via heuristic text fallback."}
        
        # Remove or comment out this block:
        # elif "VALID" in upper_text or "BENIGN" in upper_text or "SAFE" in upper_text:
        #     return {"action": "VALID", "confidence": 0.75, "reasoning": "Parsed via heuristic text fallback."}

        return default_fail

    def _node_evaluate_threat(self, state: AgentState) -> Dict[str, Any]:
        start_wall, start_cpu = perf_counter(), process_time()
        sig = state["signal"]
        risk = getattr(sig, "risk_score", 0.50)
        entropy = compute_shannon_entropy(risk)
        prompt = state.get("formatted_prompt", "")

        try:
            response = self.llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)

            parsed = self._extract_action_from_json(raw_text)
            proposed_action = str(parsed.get("action", "")).strip().upper()

            if proposed_action in ["VALID", "SPAM", "PHISHING", "QUARANTINE"]:
                final_action = proposed_action
                reasoning = parsed.get("reasoning", "Escalated to LLM due to threat uncertainty.")
            else:
                final_action = state["signal"].context.get("ml_predicted_class", "QUARANTINE")
                reasoning = f"Unresolved output; mapped to ML policy ({final_action})."

            # --- POST-LLM OVERRIDE INJECTION ---
            high_bound = getattr(cfg.conformal, "high_bound", 0.95)
            max_risk = sig.context.get("max_risk_score", risk)
            raw_text_lower = sig.raw_payload.lower()

            # Indicator 1: Active URLs
            has_external_links = "http://" in raw_text_lower or "https://" in raw_text_lower

            # Indicator 2: SOP-104 Social Engineering Vectors (Linkless Scams)
            social_eng_keywords = [
                "practicum", "incubator cohort", "bootcamp", "gift card", 
                "claim code", "apple gift", "reimburse", "board meeting"
            ]
            has_social_eng_triggers = any(kw in raw_text_lower for kw in social_eng_keywords)

            # Post-LLM Override Logic
            if final_action == "VALID" and max_risk >= high_bound:
                if has_external_links or has_social_eng_triggers:
                    final_action = "QUARANTINE"
                    reasoning = (
                        f"Post-LLM Override: LLM returned VALID, but high-risk threat signal "
                        f"({max_risk:.2f}) matched active URL or SOP-104 social engineering indicators."
                    )
            conf = float(parsed.get("confidence", risk))

            wall_latency = (perf_counter() - start_wall) * 1000
            cpu_latency = (process_time() - start_cpu) * 1000

            return {
                "action": final_action,
                "raw_response": raw_text,
                "reasoning": reasoning,
                "confidence": conf,
                "latency_ms": wall_latency,
                "error": None,
                "uncertainty_metadata": {
                    "evaluation_mode": "ESCALATION_RAG_LLM",
                    "self_reported_confidence": conf,
                    "risk_score": risk,
                    "max_risk_score": sig.context.get("max_risk_score", 0.0),
                    "shannon_entropy_bits": entropy,
                    "cpu_latency_ms": round(cpu_latency, 2),
                    "wall_latency_ms": round(wall_latency, 2),
                    "hardware_device": self._get_device_name()
                }
            }

        except Exception as e:
            print(f"\n[!] LLM EXECUTION ERROR: {str(e)}\n")
            wall_latency = (perf_counter() - start_wall) * 1000
            cpu_latency = (process_time() - start_cpu) * 1000

            return {
                "error": str(e),
                "latency_ms": wall_latency,
                "uncertainty_metadata": {
                    "error": str(e),
                    "cpu_latency_ms": round(cpu_latency, 2),
                    "wall_latency_ms": round(wall_latency, 2),
                    "hardware_device": self._get_device_name()
                }
            }

    def _node_execute_action(self, state: AgentState) -> Dict[str, Any]:
        sig = state["signal"]
        act = state["action"]
        confidence = state.get("confidence", 0.90)
        reasoning = state.get("reasoning", "Action processed.")

        if act == 'PHISHING':
            audit_id = self.tools.block(sig.signal_id, f"[{confidence:.2f}] PHISHING: {reasoning}")
        elif act == 'SPAM':
            audit_id = self.tools.quarantine(sig.signal_id, f"[{confidence:.2f}] SPAM: {reasoning}")
        elif act == 'VALID':
            audit_id = self.tools.allow(sig.signal_id, confidence)
        else:
            audit_id = self.tools.quarantine(sig.signal_id, f"Default Quarantine: {reasoning}")

        return {"audit_id": audit_id}

    def _node_fallback_handler(self, state: AgentState) -> Dict[str, Any]:
        sig = state["signal"]
        err = state.get("error", "Unknown error")
        reason = f"Fallback quarantine due to execution error: {err}"
        audit_id = self.tools.quarantine(sig.signal_id, reason)

        return {
            "action": "QUARANTINE",
            "reasoning": reason,
            "audit_id": audit_id,
            "confidence": 0.0,
            "uncertainty_metadata": {
                "error": err,
                "evaluation_mode": "fallback_handler",
                "hardware_device": self._get_device_name()
            }
        }

    def respond(self, signal: ThreatSignal, adapter: DomainAdapter) -> DecisionOutput:
        initial_state = {
            "signal": signal,
            "adapter": adapter,
            "formatted_prompt": "",
            "raw_response": "",
            "action": "",
            "reasoning": "",
            "audit_id": "",
            "latency_ms": 0.0,
            "error": None,
            "confidence": 0.5,
            "uncertainty_metadata": {}
        }
        final_state = self.graph.invoke(initial_state)
        return DecisionOutput(
            action=final_state["action"],
            reasoning=final_state["reasoning"],
            audit_id=final_state["audit_id"],
            latency_ms=final_state["latency_ms"],
            confidence=final_state.get("confidence"),
            metadata=final_state.get("uncertainty_metadata", {})
        )