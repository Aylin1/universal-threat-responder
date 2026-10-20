from types import SimpleNamespace

from adapters.rag_spam_adapter import (
    RAGSpamAdapter,
    compute_shannon_entropy,
    email_contains_indicator,
    l2_to_cosine_similarity,
)
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
    assert context["retrieval_confidence"] == 0.975


def test_l2_distance_converts_to_cosine_similarity():
    assert l2_to_cosine_similarity(0.0) == 1.0
    assert l2_to_cosine_similarity(1.0) == 0.5
    assert l2_to_cosine_similarity(2.0) == 0.0
    assert l2_to_cosine_similarity(4.0) == 0.0


def test_adapter_returns_empty_retrieval_defaults():
    context = make_adapter([]).build_context("test signal")

    assert context["retrieved_matches"] == []
    assert context["max_risk_score"] == 0.0
    assert context["high_risk_hits"] == 0
    assert context["retrieval_confidence"] == 0.0


def test_prompt_includes_retrieved_matches_and_policies():
    adapter = make_adapter([])
    signal = SimpleNamespace(
        raw_payload="hello",
        risk_score=0.4,
        context={
            "raw_input": "Verify your account now",
            "retrieved_matches": [
                {"Source": "OpenPhish", "Mapped SOP": "SOP-103", "Category": "phishing_link",
                 "Risk Score": 0.97, "Indicator Signature": "http://evil.invalid/login", "Vector Distance": 0.3},
            ],
        },
    )

    text = adapter.get_prompt_template()(signal, adapter)

    assert "http://evil.invalid/login" in text
    assert "OpenPhish" in text and "SOP-103" in text
    assert "SOP-102" in text


def test_prompt_handles_missing_retrieval():
    adapter = make_adapter([])
    signal = SimpleNamespace(raw_payload="hi", risk_score=0.5, context={})

    assert "No relevant threat intelligence" in adapter.get_prompt_template()(signal, adapter)


def test_entropy_is_finite_and_maximal_at_uncertainty():
    assert compute_shannon_entropy(0.5) == 1.0
    assert 0.0 <= compute_shannon_entropy(0.0) < 0.001
    assert 0.0 <= compute_shannon_entropy(1.0) < 0.001


def test_build_context_searches_with_condensed_query():
    queries = []
    adapter = RAGSpamAdapter.__new__(RAGSpamAdapter)
    adapter.vector_store = SimpleNamespace(
        similarity_search_with_score=lambda query, k: queries.append(query) or []
    )

    adapter.build_context("Subject: Verify now\n\nVisit https://evil.invalid/login today")

    assert queries[0].startswith("Subject: Verify now | https://evil.invalid/login")


def test_relevance_cutoff_comes_from_config(monkeypatch):
    results = [(SimpleNamespace(page_content="Newsletter", metadata={}), 0.5)]  # cosine 0.75

    monkeypatch.setattr(cfg.vector_store, "min_similarity_cutoff", 0.80)
    assert make_adapter(results).build_context("x")["retrieved_matches"] == []

    monkeypatch.setattr(cfg.vector_store, "min_similarity_cutoff", 0.70)
    assert len(make_adapter(results).build_context("x")["retrieved_matches"]) == 1


IOC_RECORD = "Threat Vector: Active Phishing URL / Link\nTarget URL: http://evil.invalid/login\nSource: OpenPhish Live Feed"


def test_email_contains_indicator_matches_normalized_urls():
    assert email_contains_indicator("Please visit http://evil.invalid/login.", IOC_RECORD)
    assert email_contains_indicator("Go to https://www.EVIL.invalid/login/ now", IOC_RECORD)
    assert not email_contains_indicator("Go to http://evil.invalid/other", IOC_RECORD)
    assert not email_contains_indicator("http://evil.invalid/login", "Subject: no indicator here")


def test_build_context_reports_ioc_match_and_high_risk_similarity():
    adapter = make_adapter([
        (SimpleNamespace(page_content=IOC_RECORD, metadata={"risk_score": 0.95}), 0.05),
        (SimpleNamespace(page_content="Newsletter", metadata={"risk_score": 0.40}), 0.30),
    ])

    context = adapter.build_context("Please visit http://evil.invalid/login")

    assert context["ioc_match"] is True
    assert context["high_risk_similarity"] == 0.975


def test_build_context_without_indicator_has_no_ioc_match():
    adapter = make_adapter([
        (SimpleNamespace(page_content=IOC_RECORD, metadata={"risk_score": 0.95}), 0.60),
    ])

    context = adapter.build_context("Lunch tomorrow? No links here.")

    assert context["ioc_match"] is False
    assert context["high_risk_similarity"] == 0.70

