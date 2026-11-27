import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pymupdf
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import adapters.phishfuzzer_loader
import adapters.rag_spam_adapter
import langchain_ollama
import rapidocr_onnxruntime
import src.core


class Upload(io.BytesIO):
    def __init__(self, name, content):
        super().__init__(content)
        self.name = name

@pytest.fixture
def dashboard(monkeypatch):
    st.cache_resource.clear()
    ocr = MagicMock()
    classifier = MagicMock()
    classifier.classify_text.return_value = {
        "ml_predicted_class": "SPAM",
        "ml_probs": {"VALID": 0.1, "SPAM": 0.8, "PHISHING": 0.1},
        "conformal_set": ["SPAM"],
    }
    adapter = MagicMock()
    adapter.build_context.return_value = {}
    graph = MagicMock()
    graph.respond.return_value = SimpleNamespace(
        action="SPAM", metadata={}, latency_ms=1, reasoning="test", audit_id="test"
    )
    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", lambda: ocr)
    monkeypatch.setattr(adapters.phishfuzzer_loader, "load_classifier", lambda _: classifier)
    monkeypatch.setattr(adapters.rag_spam_adapter, "RAGSpamAdapter", lambda **_: adapter)
    monkeypatch.setattr(langchain_ollama, "ChatOllama", lambda **_: None)
    monkeypatch.setattr(src.core, "SecurityTools", lambda **_: None)
    monkeypatch.setattr(src.core, "ThreatResponderGraph", lambda **_: graph)
    return ocr, classifier, adapter


def run_upload(monkeypatch, upload):
    def file_uploader(label, **kwargs):
        if kwargs["key"] == "pdf_uploader":
            assert "png" in kwargs["type"] and "pdf" in kwargs["type"]
            return upload
        return None

    monkeypatch.setattr(st, "file_uploader", file_uploader)
    return AppTest.from_file(Path(__file__).resolve().parent.parent / "app.py", default_timeout=30).run()


def test_png_ocr_text_is_analyzed(monkeypatch, dashboard):
    ocr, classifier, adapter = dashboard
    ocr.return_value = ([[[], "Suspicious message", 0.99], [[], "Click here", 0.9]], None)
    app = run_upload(monkeypatch, Upload("message.PNG", b"png bytes"))

    assert not app.exception
    ocr.assert_called_once_with(b"png bytes")
    classifier.classify_text.assert_called_once_with("Suspicious message\nClick here")
    assert adapter.build_context.call_args.kwargs["metadata"]["source"] == "png_upload"
    assert app.session_state["last_analyzed_upload"]


@pytest.mark.parametrize("error", [False, True])
def test_png_ocr_failure_does_not_analyze(monkeypatch, dashboard, error):
    ocr, classifier, _ = dashboard
    if error:
        ocr.side_effect = ValueError("invalid image")
    else:
        ocr.return_value = (None, None)
    app = run_upload(monkeypatch, Upload("message.png", b"invalid"))

    assert not app.exception
    expected_error = "invalid image" if error else "failed to detect text"
    assert any(expected_error in e.value for e in app.error)
    classifier.classify_text.assert_not_called()


def test_pdf_upload_still_uses_digital_text(monkeypatch, dashboard):
    ocr, classifier, adapter = dashboard
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Digital PDF text")
        pdf = document.tobytes()
    app = run_upload(monkeypatch, Upload("message.pdf", pdf))

    assert not app.exception
    ocr.assert_not_called()
    classifier.classify_text.assert_called_once_with("Digital PDF text")
    assert adapter.build_context.call_args.kwargs["metadata"]["source"] == "pdf_upload"
