from types import SimpleNamespace

from adapters.rag_spam_adapter import RAGSpamAdapter, compute_shannon_entropy
from config import cfg


def make_adapter(search_results):
    adapter = RAGSpamAdapter.__new__(RAGSpamAdapter)
    adapter.vector_store = SimpleNamespace(
        similarity_search_with_score=lambda query, k: search_results
    )
    return adapter


def test_adapter_builds_context_filters_irrelevant_and_maps_sop(monkeypatch):
    monkeypatch.setattr(cfg.risk, "high_risk_threshold", 0.95)
    adapter = make_adapter([
        (
            SimpleNamespace(
                page_content="Login at https://example.invalid",
                metadata={"risk_score": 0.98, "source": "unit-test"},
            ),
            0.05,
        ),
        (
            SimpleNamespace(
                page_content="Unsolicited bulk email",
                metadata={"risk_score": 0.65},
            ),
            0.25,
        ),
        (
            SimpleNamespace(page_content="Irrelevant", metadata={}),
            0.95,
        ),
    ])

    context = adapter.build_context("test signal")

    assert len(context["retrieved_matches"]) == 2
    assert context["max_risk_score"] == 0.98
    assert context["max_risk_sop"] == "SOP-103"
    assert context["high_risk_hits"] == 1
    assert context["retrieved_matches"][1]["Mapped SOP"] == "SOP-101"
    assert context["retrieval_confidence"] == 0.95


def test_adapter_returns_empty_retrieval_defaults():
    context = make_adapter([]).build_context("test signal")

    assert context["retrieved_matches"] == []
    assert context["max_risk_score"] == 0.0
    assert context["high_risk_hits"] == 0
    assert context["retrieval_confidence"] == 0.0


def test_entropy_is_finite_and_maximal_at_uncertainty():
    assert compute_shannon_entropy(0.5) == 1.0
    assert 0.0 <= compute_shannon_entropy(0.0) < 0.001
    assert 0.0 <= compute_shannon_entropy(1.0) < 0.001
