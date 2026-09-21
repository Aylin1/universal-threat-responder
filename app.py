import io
import random
import streamlit as st

import pymupdf as fitz  
import pdfplumber
from pypdf import PdfReader

# OCR Engine Import
try:
    from rapidocr_onnxruntime import RapidOCR
    rapid_ocr = RapidOCR()
except ImportError:
    rapid_ocr = None

from langchain_ollama import ChatOllama
from adapters.rag_spam_adapter import RAGSpamAdapter
from src.core import ThreatResponderGraph, SecurityTools
from src.domain_interface import ThreatSignal
from config import cfg

st.set_page_config(page_title="Universal Threat Responder", page_icon="🛡️", layout="wide")

st.header("🛡️ Universal Threat Responder: Rapid Prototyping Dashboard")
st.markdown("Test your threat intelligence and classification engine interactively with raw text, uploaded PDF documents, or text files.")

@st.cache_resource
def load_threat_engine():
    adapter = RAGSpamAdapter(
        milvus_uri=cfg.vector_store.uri,
        collection_name=getattr(cfg.vector_store, "phishing_collection", "phishing_threat_intel")
    )
    llm = ChatOllama(
        model=cfg.llm.model_name, 
        temperature=cfg.llm.temperature, 
        keep_alive=cfg.llm.keep_alive
    )
    tools = SecurityTools(audit_log_path=cfg.paths.audit_log_path)
    graph = ThreatResponderGraph(
        llm=llm,
        tools=tools,
        confidence_threshold=cfg.llm.confidence_threshold,
        use_fast_path=cfg.conformal.use_fast_path
    )
    return adapter, graph

def extract_pdf_text_diagnostic(file_bytes: bytes) -> str:
    """Multi-engine PDF text extractor with automatic OCR fallback for image/outlined PDFs."""
    st.write("### 🛠️ PDF Extraction Pipeline")
    
    # Engine 1: PyMuPDF Digital Text
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        extracted = [page.get_text("text") for page in doc if page.get_text("text").strip()]
        if extracted:
            st.success("✅ Digital Text Extracted via PyMuPDF!")
            return "\n".join(extracted).strip()
    except Exception as e:
        st.error(f"❌ PyMuPDF Engine Error: {e}")

    # Engine 2: pdfplumber Digital Text
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            extracted = [p.extract_text() for p in pdf.pages if p.extract_text()]
            if extracted:
                st.success("✅ Digital Text Extracted via pdfplumber!")
                return "\n".join(extracted).strip()
    except Exception:
        pass

    # Engine 3: pypdf Digital Text
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted = [page.extract_text() for page in reader.pages if page.extract_text()]
        if extracted:
            st.success("✅ Digital Text Extracted via pypdf!")
            return "\n".join(extracted).strip()
    except Exception:
        pass

    # Engine 4: OCR Fallback for Scanned/Vector-Outlined PDFs
    st.info("ℹ️ No digital text streams found. Initiating Optical Character Recognition (OCR)...")
    
    if rapid_ocr is not None:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            ocr_text_pages = []
            
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                result, _ = rapid_ocr(img_bytes)
                if result:
                    page_text = "\n".join([line[1] for line in result])
                    ocr_text_pages.append(page_text)
            
            if ocr_text_pages:
                st.success("✅ OCR Engine successfully extracted text from visual PDF layers!")
                return "\n".join(ocr_text_pages).strip()
            else:
                st.error("❌ OCR failed to detect text in the rendered pages.")
        except Exception as ocr_err:
            st.error(f"❌ OCR Extraction Error: {ocr_err}")
    else:
        st.warning("⚠️ OCR engine not installed. Run `pip install rapidocr-onnxruntime` to process image/scanned PDFs.")

    return ""

with st.spinner("Initializing Milvus vector database and local LLM graph..."):
    adapter, agent = load_threat_engine()

# Sidebar Telemetry View
st.sidebar.header("Engine Telemetry Configuration")
st.sidebar.text(f"Fast-Path Enabled: {cfg.conformal.use_fast_path}")
st.sidebar.text(f"Conformal Bounds: [{cfg.conformal.low_bound}, {cfg.conformal.high_bound}]")
st.sidebar.text(f"Model: {cfg.llm.model_name}")

email_text = ""

# Input Navigation Tabs
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📝 Text Paste", 
    "📄 PDF Document Upload", 
    "📁 Text File Upload"
])

with input_tab1:
    pasted_text = st.text_area("Paste raw email or threat payload text here:", height=250, key="pasted_input")
    if pasted_text:
        email_text = pasted_text.strip()

with input_tab2:
    uploaded_pdf = st.file_uploader("Upload threat report or email PDF", type=["pdf"], key="pdf_uploader")
    if uploaded_pdf is not None:
        raw_bytes = uploaded_pdf.getvalue()
        email_text = extract_pdf_text_diagnostic(raw_bytes)
        if not email_text:
            st.error("⚠️ Extraction Failed: Document text could not be extracted.")

with input_tab3:
    uploaded_txt = st.file_uploader("Upload plain text file", type=["txt"], key="txt_uploader")
    if uploaded_txt is not None:
        try:
            email_text = uploaded_txt.read().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            st.error(f"Error reading text file: {e}")

st.markdown("---")
submitted = st.button("🚀 Analyze Threat Signal", type="primary")

# Execution block after button click
if submitted:
    if not email_text:
        st.warning("Please provide input text, upload a valid PDF document, or upload a text file first.")
    else:
        with st.spinner("Executing RAG retrieval and decision graph workflow..."):
            context = adapter.build_context(
                raw_input=email_text,
                metadata={"source": "streamlit_ui"}
            )
            
            signal = ThreatSignal(
                signal_id=f"ST-{random.randint(100000, 999999)}",
                domain="spam_phishing",
                raw_payload=email_text,
                risk_score=context.get("max_risk_score", 0.50),
                confidence=0.85,
                context=context
            )
            
            decision = agent.respond(signal=signal, adapter=adapter)
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="Decision Action", value=decision.action)
            with col2:
                st.metric(label="Confidence Score", value=f"{decision.confidence:.2%}" if decision.confidence else "N/A")
            with col3:
                st.metric(label="Max Threat Risk", value=f"{context.get('max_risk_score', 0.0):.2f}")
            with col4:
                st.metric(label="Latency", value=f"{decision.latency_ms:.2f} ms")
            
            st.subheader("🔍 Agent Reasoning & Audit Trace")
            st.info(decision.reasoning)
            
            with st.expander("View Detailed Telemetry & Retrieved Threat Signatures"):
                st.write(f"**Audit ID:** `{decision.audit_id}`")
                st.write(f"**High-Risk Threat Matches:** `{context.get('high_risk_hits', 0)}`")
                
                st.write("**Retrieved Threat Intelligence Indicators:**")
                retrieved_matches = context.get("retrieved_matches", [])
                if retrieved_matches:
                    st.dataframe(retrieved_matches)
                else:
                    st.write("No matching threat intelligence signatures retrieved.")
                
                st.write("**Evaluated Standard Operating Procedures (SOPs):**")
                st.write(context.get("retrieved_policies", []))
                
                st.write("**Raw Decision Metadata:**")
                st.json(decision.metadata)