import math
import numpy as np
from typing import Dict, Any
from adapters.rag_spam_adapter import compute_shannon_entropy
from src.domain_interface import ThreatSignal


class DetailedThreatEvaluator:

    def __init__(
        self, 
        temperature_tau: float = 0.2, 
        min_similarity_cutoff: float = 0.35,
        high_bound: float = 0.95,
        low_bound: float = 0.05
    ):
        self.tau = temperature_tau
        self.cutoff = min_similarity_cutoff
        self.high_bound = high_bound
        self.low_bound = low_bound
        self.epistemic_entropy_threshold = compute_shannon_entropy(self.low_bound)

    def compute_ml_uncertainty(self, risk_score: float) -> Dict[str, float]:
        p = float(np.clip(risk_score, 1e-12, 1.0 - 1e-12))
        entropy_bits = compute_shannon_entropy(p)
        return {
            "ml_risk_raw_probability": p,
            "ml_shannon_entropy_bits": entropy_bits,
            "ml_margin_ambiguity": float(1.0 - 2.0 * abs(p - 0.5)),
        }

    def compute_rag_uncertainty(self, context: Dict[str, Any]) -> Dict[str, float]:
        retrieval_confidence = float(context.get("retrieval_confidence", 0.0))
        used_fallback = context.get("used_fallback_sop", False)

        if used_fallback or retrieval_confidence == 0.0:
            return {
                "rag_top1_similarity": 0.0,
                "rag_top1_gap": 1.0,
                "rag_softmax_entropy_bits": 0.0,
                "rag_below_cutoff_penalty": 1.0,
            }

        top1_gap = max(0.0, 1.0 - retrieval_confidence)
        
        # Guard against zero-division if cutoff is set to 0.0
        if self.cutoff > 0.0:
            cutoff_penalty = float(max(0.0, (self.cutoff - retrieval_confidence) / self.cutoff))
        else:
            cutoff_penalty = 0.0

        return {
            "rag_top1_similarity": retrieval_confidence,
            "rag_top1_gap": top1_gap,
            "rag_softmax_entropy_bits": compute_shannon_entropy(retrieval_confidence),
            "rag_below_cutoff_penalty": cutoff_penalty,
        }

    def compute_structural_hallucinations(self, llm_reasoning: str, context: Dict[str, Any]) -> Dict[str, Any]:
        reasoning_lower = llm_reasoning.lower()
        claims_checked = 1
        contradictions = 0
        hallucination_events = []

        risk_score = context.get("ml_risk_raw_probability", context.get("risk_score", 0.5))
        high_risk_hits = context.get("high_risk_hits", 0)

        # Contradiction check: LLM claiming benign status on active high-risk hits
        if any(w in reasoning_lower for w in ["completely safe", "entirely legitimate", "no threat detected", "valid business"]):
            if risk_score > self.high_bound or high_risk_hits > 0:
                contradictions += 1
                hallucination_events.append(f"LLM claimed safety on verified {context.get('threat_source', 'known threat intelligence')} threat signal.")

        hallucination_rate = float(contradictions / claims_checked) if claims_checked > 0 else 0.0

        return {
            "total_claims_validated": claims_checked,
            "contradiction_count": contradictions,
            "structural_hallucination_rate": hallucination_rate,
            "hallucination_details": hallucination_events,
        }

    def evaluate_signal(
        self,
        signal: ThreatSignal,
        llm_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctx = signal.context or {}

        ml_telemetry = self.compute_ml_uncertainty(signal.risk_score)
        rag_telemetry = self.compute_rag_uncertainty(ctx)
        hallucination_telemetry = self.compute_structural_hallucinations(
            llm_response.get("reasoning", ""), ctx
        )

        llm_confidence = float(np.clip(llm_response.get("confidence", 0.0), 0.0, 1.0))
        expected_confidence = float(0.50 + abs(signal.risk_score - 0.50))
        calibration_error = float(abs(llm_confidence - expected_confidence))

        return {
            "signal_id": signal.signal_id,
            "domain": signal.domain,
            "ml_uncertainty": ml_telemetry,
            "rag_uncertainty": rag_telemetry,
            "hallucination_metrics": hallucination_telemetry,
            "llm_calibration": {
                "llm_verbalized_confidence": llm_confidence,
                "expected_model_confidence": expected_confidence,
                "verbalization_calibration_error": calibration_error,
            },
            "vector_flags": {
                "high_epistemic_uncertainty": ml_telemetry["ml_shannon_entropy_bits"] > self.epistemic_entropy_threshold,
                "high_retrieval_uncertainty": rag_telemetry["rag_below_cutoff_penalty"] > 0.0,
                "structural_hallucination_detected": hallucination_telemetry["contradiction_count"] > 0,
                "overconfident_llm": (llm_confidence - expected_confidence) > 0.20,
            },
        }