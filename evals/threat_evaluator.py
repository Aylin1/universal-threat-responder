"""Per-signal telemetry.

Renamed/replaced because the old names claimed more than the code measured:

* "verbalization_calibration_error" compared the LLM's stated confidence with a function of the
  ML probability and never looked at correctness. It is now `confidence_gap_vs_ml` (descriptive
  only). Real calibration (ECE / Brier against correctness) is computed in evals/analyze_results.py.
* "shannon_entropy" was the BINARY entropy of the top-class probability. It is now the entropy
  of the full 3-class predictive distribution (`ml_entropy_bits`) plus the top-2 margin.
* "structural_hallucination_rate" was a 4-phrase keyword check with claims_checked fixed at 1.
  It is split into (a) `unsupported_citations`: reasoning cites an SOP/indicator id that was
  never provided to the model (a genuine, checkable hallucination) and (b)
  `safe_claim_contradiction`: reasoning asserts safety while hard evidence says otherwise.
"""
import re
from typing import Any, Dict, Optional

import numpy as np

from src.domain_interface import ThreatSignal
from src.metrics import compute_shannon_entropy, distribution_entropy, top2_margin

_CITATION_RE = re.compile(r"\b(?:SOP-\d{3}|SOCENG-[A-Z]+-\d+|SPAM-[A-Z]+-\d+|OPENPHISH-\d+|URLHAUS-\d+)\b")
_SAFE_PHRASES = ("completely safe", "entirely legitimate", "no threat detected", "valid business")


class DetailedThreatEvaluator:
    def __init__(
        self,
        temperature_tau: float = 0.2,          # unused; kept so existing call sites don't break
        min_similarity_cutoff: float = 0.35,
        high_bound: float = 0.95,              # deprecated: routing is conformal now
        low_bound: float = 0.05,               # deprecated
        high_entropy_bits: float = 0.80,
        overconfidence_margin: float = 0.20,
    ):
        self.cutoff = min_similarity_cutoff
        self.high_entropy_bits = high_entropy_bits
        self.overconfidence_margin = overconfidence_margin

    # ------------------------------------------------------------------ pieces
    def compute_ml_uncertainty(self, signal: ThreatSignal) -> Dict[str, float]:
        probs: Optional[Dict[str, float]] = (signal.context or {}).get("ml_probs")
        if probs:
            vals = list(probs.values())
            return {"ml_top_prob": float(max(vals)),
                    "ml_entropy_bits": distribution_entropy(vals),
                    "ml_top2_margin": top2_margin(vals)}
        p = float(np.clip(signal.risk_score, 1e-12, 1 - 1e-12))
        return {"ml_top_prob": p, "ml_entropy_bits": compute_shannon_entropy(p),
                "ml_top2_margin": float("nan")}

    def compute_rag_telemetry(self, ctx: Dict[str, Any]) -> Dict[str, float]:
        conf = float(ctx.get("retrieval_confidence", 0.0))
        penalty = max(0.0, (self.cutoff - conf) / self.cutoff) if self.cutoff > 0 else 0.0
        return {"rag_top1_similarity": conf,
                "rag_below_cutoff_penalty": float(penalty),
                "nn_top_sim": float(ctx.get("nn_top_sim", 0.0))}

    def compute_grounding(self, reasoning: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        text = reasoning or ""
        allowed = set(ctx.get("allowed_citation_ids", []))
        cited = set(_CITATION_RE.findall(text))
        # Only judge citations when we know what was provided; otherwise we can't call them unsupported.
        unsupported = sorted(cited - allowed) if allowed else []

        lower = text.lower()
        claims_safe = any(p in lower for p in _SAFE_PHRASES)
        hard_evidence = int(ctx.get("ioc_hits", 0)) > 0
        contradiction = bool(claims_safe and hard_evidence)

        return {"unsupported_citations": unsupported,
                "n_citations": len(cited),
                "safe_claim_contradiction": contradiction}

    # ------------------------------------------------------------------ main
    def evaluate_signal(self, signal: ThreatSignal, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        ctx = signal.context or {}
        ml = self.compute_ml_uncertainty(signal)
        rag = self.compute_rag_telemetry(ctx)
        grounding = self.compute_grounding(llm_response.get("reasoning", ""), ctx)

        llm_conf = float(np.clip(llm_response.get("confidence", 0.0), 0.0, 1.0))
        gap = llm_conf - ml["ml_top_prob"]

        return {
            "signal_id": signal.signal_id,
            "domain": signal.domain,
            "ml_uncertainty": ml,
            "rag": rag,
            "grounding": grounding,
            "confidence": {"llm_verbalized_confidence": llm_conf, "confidence_gap_vs_ml": gap},
            "flags": {
                "high_ml_entropy": ml["ml_entropy_bits"] > self.high_entropy_bits,
                "weak_retrieval": rag["rag_below_cutoff_penalty"] > 0.0,
                "unsupported_citation": len(grounding["unsupported_citations"]) > 0,
                "safe_claim_contradiction": grounding["safe_claim_contradiction"],
                "llm_more_confident_than_ml": gap > self.overconfidence_margin,
            },
        }
