import json
import logging
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Dict, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from src.config_access import cfg_get
from src.domain_interface import ThreatSignal, DecisionOutput, DomainAdapter
from src.guard import evaluate_agreement_review
from src.metrics import compute_shannon_entropy, distribution_entropy

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


# --------------------------------------------------------------------------- routing
def route_threat_signal(
    state: AgentState,
    high_accuracy_bound: Optional[float] = None,
    use_fast_path: bool = True,
) -> str:
    """Decide the route for a signal.

    Order of precedence:
      1. deterministic IOC hit (exact URL in a threat feed)      -> "ioc_block"
      2. conformal singleton prediction set (if the loader supplied one) -> "fast_path_bypass"
      3. legacy fallback: risk_score >= high_bound (only when no conformal info is present)
      4. otherwise                                               -> "escalate_to_llm"

    The IOC route applies even when the fast path is disabled, because it is a deterministic
    rule and not a model decision.
    """
    sig = state["signal"]
    ctx = sig.context or {}

    if any(m.get("match_type") == "url" for m in ctx.get("ioc_matches", []) or []):
        return "ioc_block"

    if not use_fast_path:
        return "escalate_to_llm"

    if "conformal_singleton" in ctx:
        return "fast_path_bypass" if ctx["conformal_singleton"] else "escalate_to_llm"

    max_prob = getattr(sig, "risk_score", 0.50)
    hb = (cfg_get("conformal", "high_bound", 0.95)
          if high_accuracy_bound is None else high_accuracy_bound)
    return "fast_path_bypass" if max_prob >= hb else "escalate_to_llm"


class ThreatResponderGraph:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Optional[SecurityTools] = None,
        uncertainty_samples: int = 1,
        confidence_threshold: float = 0.50,
        use_fast_path: bool = True,
        require_agreement_review: bool = False,
    ):
        """
        uncertainty_samples: number of LLM draws for self-consistency voting. Only
            informative when the LLM temperature > 0 (at temperature 0 all draws agree).
        confidence_threshold: a verdict whose confidence falls below this abstains
            (fallback action). Verbalised confidence is weak evidence, so keep it modest.
        """
        self.llm = llm
        self.tools = tools or SecurityTools()
        self.uncertainty_samples = max(1, int(uncertainty_samples))
        self.confidence_threshold = float(confidence_threshold)
        self.use_fast_path = use_fast_path
        self.require_agreement_review = require_agreement_review
        if require_agreement_review and use_fast_path:
            raise ValueError("Agreement review requires the fast path to be disabled")
        self.graph = self._build_graph()

    def _get_device_name(self) -> str:
        try:
            import torch
            return "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        except ImportError:
            return "CPU"

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("ioc_block", self._node_ioc_block)
        builder.add_node("fast_path_bypass", self._node_fast_path_bypass)
        builder.add_node("build_prompt", self._node_build_prompt)
        builder.add_node("evaluate_threat", self._node_evaluate_threat)
        builder.add_node("execute_action", self._node_execute_action)
        builder.add_node("fallback_handler", self._node_fallback_handler)

        use_fp = self.use_fast_path

        def _route(state: AgentState) -> str:
            if self.require_agreement_review:
                return "escalate_to_llm"
            return route_threat_signal(state, use_fast_path=use_fp)

        builder.add_conditional_edges(
            START, _route,
            {
                "ioc_block": "ioc_block",
                "fast_path_bypass": "fast_path_bypass",
                "escalate_to_llm": "build_prompt",
            },
        )
        builder.add_edge("ioc_block", "execute_action")
        builder.add_edge("fast_path_bypass", "execute_action")
        builder.add_edge("build_prompt", "evaluate_threat")
        builder.add_conditional_edges(
            "evaluate_threat",
            lambda state: "fallback_handler" if state.get("error") else "execute_action",
            {"fallback_handler": "fallback_handler", "execute_action": "execute_action"},
        )
        builder.add_edge("execute_action", END)
        builder.add_edge("fallback_handler", END)
        return builder.compile()

    # ------------------------------------------------------------------ nodes
    def _node_ioc_block(self, state: AgentState) -> Dict[str, Any]:
        start_wall = perf_counter()
        sig, adapter = state["signal"], state["adapter"]
        hits = [m for m in sig.context.get("ioc_matches", []) if m.get("match_type") == "url"]
        ids = ", ".join(sorted({m.get("indicator_id") or m.get("source", "?") for m in hits}))
        reasoning = f"Deterministic IOC match: {len(hits)} URL(s) present in threat feed ({ids})."
        wall = (perf_counter() - start_wall) * 1000
        return {
            "action": adapter.ioc_block_action,
            "raw_response": json.dumps({"action": adapter.ioc_block_action, "reasoning": reasoning}),
            "reasoning": reasoning,
            "confidence": 1.0,
            "latency_ms": wall,
            "error": None,
            "uncertainty_metadata": {
                "evaluation_mode": "IOC_BLOCK",
                "ioc_hits": len(hits),
                "wall_latency_ms": round(wall, 2),
                "hardware_device": self._get_device_name(),
            },
        }

    def _node_fast_path_bypass(self, state: AgentState) -> Dict[str, Any]:
        start_wall, start_cpu = perf_counter(), process_time()
        sig, adapter = state["signal"], state["adapter"]
        confidence = getattr(sig, "risk_score", 0.50)
        probs = sig.context.get("ml_probs")
        entropy = (distribution_entropy(probs.values()) if probs
                   else compute_shannon_entropy(confidence))

        cset = sig.context.get("conformal_set")
        pred_class = cset[0] if sig.context.get("conformal_singleton") and len(cset or []) == 1 \
            else sig.context.get("ml_predicted_class", adapter.fallback_action)
        basis = (f"conformal singleton set {{{', '.join(cset)}}}" if cset
                 else f"calibrated probability {confidence:.4f}")
        reasoning = f"Fast-path classified as {pred_class} via {basis} (p={confidence:.4f})."
        raw_resp = json.dumps({"action": pred_class, "confidence": confidence, "reasoning": reasoning})

        wall = (perf_counter() - start_wall) * 1000
        cpu = (process_time() - start_cpu) * 1000
        return {
            "action": pred_class,
            "raw_response": raw_resp,
            "reasoning": reasoning,
            "confidence": confidence,
            "latency_ms": wall,
            "error": None,
            "uncertainty_metadata": {
                "evaluation_mode": "FAST_PATH_BYPASS",
                "risk_score": confidence,
                "ml_entropy_bits": entropy,
                "cpu_latency_ms": round(cpu, 2),
                "wall_latency_ms": round(wall, 2),
                "hardware_device": self._get_device_name(),
                # NOTE: this latency covers only this node. Retrieval/embedding happen
                # before the graph and are NOT included.
                "latency_scope": "graph_node_only",
            },
        }

    def _node_build_prompt(self, state: AgentState) -> Dict[str, Any]:
        adapter, signal = state["adapter"], state["signal"]
        template = adapter.get_prompt_template()
        prompt_text = template(signal, adapter) if callable(template) else template
        return {"formatted_prompt": prompt_text}

    def _parse_llm_text(self, raw_text: str, adapter: DomainAdapter) -> Dict[str, Any]:
        """Extract {action, confidence, reasoning}. Unparseable output ABSTAINS.

        The previous keyword fallback ("PHISH"/"BLOCK" anywhere -> PHISHING) turned
        'this is not phishing' into PHISHING. Ambiguous text now abstains and is flagged.
        """
        fail = {"action": adapter.fallback_action, "confidence": 0.0,
                "reasoning": "Failed to parse LLM response; abstaining.", "parse_failed": True,
                "confidence_reported": False}
        if not raw_text or not isinstance(raw_text, str):
            return fail
        cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return fail
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return fail
        act = parsed.get("action") or parsed.get("decision") or parsed.get("classification") or ""
        confidence_reported = True
        try:
            conf = float(parsed["confidence"])
            if not math.isfinite(conf):
                raise ValueError("non-finite confidence")
        except (KeyError, TypeError, ValueError):
            conf = 0.5
            confidence_reported = False
        return {
            "action": adapter.normalize_action(str(act)),
            "confidence": min(1.0, max(0.0, conf)),
            "reasoning": str(parsed.get("reasoning", "")) or "No reasoning provided.",
            "parse_failed": False,
            "confidence_reported": confidence_reported,
        }

    def _node_evaluate_threat(self, state: AgentState) -> Dict[str, Any]:
        start_wall, start_cpu = perf_counter(), process_time()
        sig, adapter = state["signal"], state["adapter"]
        risk = getattr(sig, "risk_score", 0.50)
        probs = sig.context.get("ml_probs")
        entropy = distribution_entropy(probs.values()) if probs else compute_shannon_entropy(risk)
        prompt = state.get("formatted_prompt", "")

        try:
            draws, raws = [], []
            for _ in range(self.uncertainty_samples):
                response = self.llm.invoke(prompt)
                raw_text = response.content if hasattr(response, "content") else str(response)
                raws.append(raw_text)
                draws.append(self._parse_llm_text(raw_text, adapter))

            votes = Counter(d["action"] for d in draws)
            top_action, top_n = votes.most_common(1)[0]
            agreement = top_n / len(draws)
            winners = [d for d in draws if d["action"] == top_action]
            conf = sum(d["confidence"] for d in winners) / len(winners)
            reasoning = winners[0]["reasoning"]
            parse_failed = all(d["parse_failed"] for d in draws)

            abstain_reason = None
            if parse_failed:
                abstain_reason = "unparseable LLM output"
            elif len(draws) > 1 and agreement <= 0.5:
                abstain_reason = f"no majority across {len(draws)} samples (agreement={agreement:.2f})"
            elif conf < self.confidence_threshold:
                abstain_reason = f"confidence {conf:.2f} below threshold {self.confidence_threshold:.2f}"

            llm_raw_action = top_action
            if abstain_reason:
                final_action = adapter.fallback_action
                reasoning = f"Abstained: {abstain_reason}."
            else:
                final_action = top_action

            final_action, guard_applied, guard_reasons = adapter.apply_policy_guard(sig, final_action)
            if guard_applied:
                reasoning = "Policy guard: " + "; ".join(guard_reasons) + f" (LLM said {llm_raw_action})."

            review_reasons = []
            if self.require_agreement_review:
                ctx = sig.context or {}
                cset = ctx.get("conformal_set")
                review = evaluate_agreement_review(
                    final_action, llm_action=llm_raw_action, ml_class=ctx.get("ml_predicted_class", ""),
                    conformal_set=cset if isinstance(cset, list) else [],
                    conformal_singleton=ctx.get("conformal_singleton") is True,
                    allowed_actions=adapter.allowed_actions,
                    quarantine_action=adapter.fallback_action, ioc_hits=ctx.get("ioc_hits", 0),
                    ioc_block_action=adapter.ioc_block_action,
                )
                final_action, review_reasons = review.action, review.reasons
                if review.applied:
                    reasoning = "Human review required: " + "; ".join(review_reasons) + "."

            wall = (perf_counter() - start_wall) * 1000
            cpu = (process_time() - start_cpu) * 1000
            return {
                "action": final_action,
                "raw_response": raws[0],
                "reasoning": reasoning,
                "confidence": conf,
                "latency_ms": wall,
                "error": None,
                "uncertainty_metadata": {
                    "evaluation_mode": "ESCALATION_RAG_LLM",
                    "llm_raw_action": llm_raw_action,
                    "self_reported_confidence": conf,
                    "llm_confidence_source": (
                        "model_reported" if all(d["confidence_reported"] for d in winners)
                        else "contains_default"
                    ),
                    "vote_agreement": agreement,
                    "llm_samples": len(draws),
                    "parse_failed": parse_failed,
                    "abstain_reason": abstain_reason,
                    "policy_guard_applied": guard_applied,
                    "policy_guard_reasons": guard_reasons,
                    "review_reasons": review_reasons,
                    "review_policies": sig.context.get("review_policies", []),
                    "risk_score": risk,
                    "ml_entropy_bits": entropy,
                    "cpu_latency_ms": round(cpu, 2),
                    "wall_latency_ms": round(wall, 2),
                    "hardware_device": self._get_device_name(),
                    "latency_scope": "graph_node_only",
                },
            }

        except Exception as e:
            logger.exception("LLM execution error")
            wall = (perf_counter() - start_wall) * 1000
            cpu = (process_time() - start_cpu) * 1000
            return {
                "error": str(e),
                "latency_ms": wall,
                "uncertainty_metadata": {
                    "error": str(e),
                    "cpu_latency_ms": round(cpu, 2),
                    "wall_latency_ms": round(wall, 2),
                    "hardware_device": self._get_device_name(),
                },
            }

    def _node_execute_action(self, state: AgentState) -> Dict[str, Any]:
        sig, adapter = state["signal"], state["adapter"]
        act = state["action"]
        confidence = state.get("confidence", 0.90)
        reasoning = state.get("reasoning", "Action processed.")

        effect = adapter.action_effects.get(act, "quarantine")
        if effect == "block":
            audit_id = self.tools.block(sig.signal_id, f"[{confidence:.2f}] {act}: {reasoning}")
        elif effect == "allow":
            audit_id = self.tools.allow(sig.signal_id, confidence)
        elif act == adapter.fallback_action:
            audit_id = self.tools.quarantine(sig.signal_id, f"Default {act}: {reasoning}")
        else:
            audit_id = self.tools.quarantine(sig.signal_id, f"[{confidence:.2f}] {act}: {reasoning}")
        return {"audit_id": audit_id}

    def _node_fallback_handler(self, state: AgentState) -> Dict[str, Any]:
        sig, adapter = state["signal"], state["adapter"]
        err = state.get("error", "Unknown error")
        reason = f"Fallback quarantine due to execution error: {err}"
        audit_id = self.tools.quarantine(sig.signal_id, reason)
        return {
            "action": adapter.fallback_action,
            "reasoning": reason,
            "audit_id": audit_id,
            "confidence": 0.0,
            "uncertainty_metadata": {
                "error": err,
                "evaluation_mode": "fallback_handler",
                "review_policies": sig.context.get("review_policies", []),
                "hardware_device": self._get_device_name(),
            },
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
            "uncertainty_metadata": {},
        }
        final_state = self.graph.invoke(initial_state)
        return DecisionOutput(
            action=final_state["action"],
            reasoning=final_state["reasoning"],
            audit_id=final_state["audit_id"],
            latency_ms=final_state["latency_ms"],
            confidence=final_state.get("confidence"),
            metadata=final_state.get("uncertainty_metadata", {}),
        )
