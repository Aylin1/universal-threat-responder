import io
import json
import csv
import hashlib
import inspect
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
    min_samples_for_fast_path = getattr(
        cfg.conformal,
        "min_samples_for_fast_path",
        5,
    )
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
    graph_kwargs = {
        "llm": llm,
        "tools": tools,
        "confidence_threshold": cfg.llm.confidence_threshold,
        "use_fast_path": cfg.conformal.use_fast_path,
    }
    if "min_samples_for_fast_path" in inspect.signature(ThreatResponderGraph).parameters:
        graph_kwargs["min_samples_for_fast_path"] = min_samples_for_fast_path

    graph = ThreatResponderGraph(
        **graph_kwargs,
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

def parse_json_samples(raw_bytes: bytes) -> list[dict]:
    data = json.loads(raw_bytes.decode("utf-8-sig"))

    if isinstance(data, dict):
        data = data.get("samples", data.get("data", [data]))

    if not isinstance(data, list):
        data = [data]

    samples = []
    for index, item in enumerate(data, start=1):
        if isinstance(item, str):
            text = item
            metadata = {}
        else:
            text = (
                item.get("text")
                or item.get("email")
                or item.get("payload")
                or item.get("raw_payload")
                or ""
            )
            metadata = item

        if str(text).strip():
            samples.append({
                "id": f"sample-{index}",
                "text": str(text).strip(),
                "metadata": metadata,
            })

    return samples


def parse_csv_samples(raw_bytes: bytes) -> list[dict]:
    content = raw_bytes.decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(io.StringIO(content))

    samples = []
    for index, row in enumerate(reader, start=1):
        text = (
            row.get("text")
            or row.get("email")
            or row.get("payload")
            or row.get("raw_payload")
            or next(iter(row.values()), "")
        )

        if str(text).strip():
            samples.append({
                "id": f"sample-{index}",
                "text": str(text).strip(),
                "metadata": dict(row),
            })

    return samples

with st.spinner("Loading threat detection model and vector store..."):
    adapter, agent = load_threat_engine()

st.sidebar.text(
    "Fast path disabled."
    if not cfg.conformal.use_fast_path
    else "Fast path enabled."
)
st.sidebar.text(f"Detection model: {cfg.llm.model_name}")

email_text = ""
samples = []

# Input Navigation Tabs
input_tab1, input_tab2, input_tab3 = st.tabs([
    "📝 Text Paste", 
    "📄 PDF Document Upload", 
    "📊 JSON / CSV / Text Batch Upload"
])

with input_tab1:
    pasted_text = st.text_area(
        "Paste raw email or threat payload text here:",
        height=250,
        key="pasted_input",
    )

    if pasted_text.strip():
        samples = [{
            "id": "pasted-input",
            "text": pasted_text.strip(),
            "metadata": {"source": "text_paste"},
        }]

with input_tab2:
    uploaded_pdf = st.file_uploader(
        "Upload threat report or email PDF",
        type=["pdf"],
        key="pdf_uploader",
    )

    if uploaded_pdf is not None:
        extracted_text = extract_pdf_text_diagnostic(uploaded_pdf.getvalue())

        if extracted_text:
            samples = [{
                "id": uploaded_pdf.name,
                "text": extracted_text,
                "metadata": {
                    "source": "pdf_upload",
                    "filename": uploaded_pdf.name,
                },
            }]
        else:
            st.error("⚠️ Could not extract text from the PDF.")

with input_tab3:
    uploaded_batch = st.file_uploader(
        "Upload JSON, CSV, or TXT file",
        type=["json", "csv", "txt"],
        key="batch_uploader",
    )

    if uploaded_batch is not None:
        raw_bytes = uploaded_batch.getvalue()
        file_type = uploaded_batch.name.lower().rsplit(".", 1)[-1]

        try:
            if file_type == "json":
                samples = parse_json_samples(raw_bytes)
            elif file_type == "csv":
                samples = parse_csv_samples(raw_bytes)
            else:
                text = raw_bytes.decode("utf-8-sig", errors="ignore").strip()
                samples = [{
                    "id": uploaded_batch.name,
                    "text": text,
                    "metadata": {
                        "source": "text_upload",
                        "filename": uploaded_batch.name,
                    },
                }] if text else []

            st.success(f"Loaded {len(samples)} sample(s).")
        except Exception as error:
            st.error(f"Failed to parse input file: {error}")

st.markdown("---")

if samples:
    st.caption(f"{len(samples)} sample(s) ready for analysis.")
    st.dataframe(
        [{"ID": item["id"], "Preview": item["text"][:150]} for item in samples],
        use_container_width=True,
    )

submitted = st.button("🚀 Analyze Threat Signal(s)", type="primary")

upload_signature = None
if uploaded_pdf is not None:
    upload_signature = hashlib.sha256(uploaded_pdf.getvalue()).hexdigest()
elif uploaded_batch is not None:
    upload_signature = hashlib.sha256(uploaded_batch.getvalue()).hexdigest()

new_upload = (
    upload_signature is not None
    and upload_signature != st.session_state.get("last_analyzed_upload")
)
analyze_requested = submitted or new_upload

if analyze_requested:
    if not samples:
        st.warning("Please provide text or upload a valid input file first.")
    else:
        results = []

        with st.spinner(f"Analyzing {len(samples)} sample(s)..."):
            for index, sample in enumerate(samples, start=1):
                try:
                    context = adapter.build_context(
                        raw_input=sample["text"],
                        metadata={
                            "source": "streamlit_ui",
                            "sample_id": sample["id"],
                            "sample_count": len(samples),
                            **sample.get("metadata", {}),
                        },
                    )

                    signal = ThreatSignal(
                        signal_id=f"ST-{random.randint(100000, 999999)}",
                        domain="spam_phishing",
                        raw_payload=sample["text"],
                        risk_score=context.get("max_risk_score", 0.50),
                        confidence=0.85,
                        context=context,
                    )

                    decision = agent.respond(signal=signal, adapter=adapter)

                    results.append({
                        "Sample": sample["id"],
                        "Action": decision.action,
                        "Confidence": decision.confidence,
                        "Risk": context.get("max_risk_score", 0.0),
                        "Latency (ms)": decision.latency_ms,
                        "Decision": decision,
                        "Context": context,
                    })

                except Exception as error:
                    results.append({
                        "Sample": sample["id"],
                        "Action": "ERROR",
                        "Confidence": 0.0,
                        "Risk": 0.0,
                        "Latency (ms)": 0.0,
                        "Error": str(error),
                    })

        st.subheader("📊 Batch Analysis Results")

        st.dataframe(
            [
                {
                    "Sample": result["Sample"],
                    "Action": result["Action"],
                    "Confidence": f'{result["Confidence"]:.2%}',
                    "Risk": f'{result["Risk"]:.2f}',
                    "Latency (ms)": f'{result["Latency (ms)"]:.2f}',
                }
                for result in results
            ],
            use_container_width=True,
        )

        for result in results:
            with st.expander(f'🔍 {result["Sample"]} — {result["Action"]}'):
                if "Error" in result:
                    st.error(result["Error"])
                    continue

                decision = result["Decision"]
                context = result["Context"]

                st.info(decision.reasoning)
                st.write(f"**Audit ID:** `{decision.audit_id}`")
                st.write(f"**High-Risk Matches:** `{context.get('high_risk_hits', 0)}`")

                st.write("**Retrieved Threat Intelligence:**")
                matches = context.get("retrieved_matches", [])
                if matches:
                    st.dataframe(matches, use_container_width=True)
                else:
                    st.write("No matching threat intelligence signatures.")

                st.write("**Retrieved Policies:**")
                st.write(context.get("retrieved_policies", []))

                st.write("**Decision Metadata:**")
                st.json(decision.metadata)

        if new_upload:
            st.session_state.last_analyzed_upload = upload_signature