from evals.threat_evaluator import DetailedThreatEvaluator
from src.domain_interface import ThreatSignal


def test_evaluator_reports_ml_rag_and_hallucination_signals():
    evaluator = DetailedThreatEvaluator(min_similarity_cutoff=0.35)
    signal = ThreatSignal(
        signal_id="evaluated",
        risk_score=0.5,
        confidence=0.8,
        context={"retrieval_confidence": 0.2, "high_risk_hits": 1},
    )

    result = evaluator.evaluate_signal(
        signal,
        {
            "confidence": 0.9,
            "reasoning": "This is entirely legitimate.",
        },
    )

    assert result["ml_uncertainty"]["ml_shannon_entropy_bits"] == 1.0
    assert result["rag_uncertainty"]["rag_below_cutoff_penalty"] > 0.0
    assert result["hallucination_metrics"]["contradiction_count"] == 1
    assert result["vector_flags"]["structural_hallucination_detected"] is True


def test_evaluator_marks_missing_retrieval_as_uncertain():
    evaluator = DetailedThreatEvaluator(min_similarity_cutoff=0.0)
    result = evaluator.compute_rag_uncertainty({})

    assert result["rag_top1_similarity"] == 0.0
    assert result["rag_top1_gap"] == 1.0
    assert result["rag_below_cutoff_penalty"] == 1.0
