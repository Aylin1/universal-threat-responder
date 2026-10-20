# 🛡️ Universal Threat Responder

### Uncertainty-aware threat triage with RAG, local LLMs, conformal routing, and live threat intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Workflow-1C3C3C)](https://www.langchain.com/langgraph)
[![Ollama](https://img.shields.io/badge/LLM-Ollama-black?logo=ollama)](https://ollama.com/)
[![Milvus](https://img.shields.io/badge/RAG-Milvus-00A1EA)](https://milvus.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

**Universal Threat Responder** is a research-oriented threat triage and decision engine combining **machine learning, retrieval-augmented generation, local LLM reasoning, conformal risk routing, and live threat intelligence**.

The system accepts heterogeneous threat signals, normalizes them into a common representation, retrieves relevant threat intelligence from a Milvus vector database, and processes the enriched signal through a stateful LangGraph decision workflow.

It supports both:

- **high-confidence fast-path decisions**, and
- **full RAG + LLM escalation** for samples requiring deeper contextual reasoning.

The project also includes an evaluation pipeline for measuring **classification performance, uncertainty, retrieval similarity, calibration, and routing behaviour**.

---

# 🔭 What This Project Demonstrates

The project focuses on the engineering problem of combining **classical ML, retrieval, local LLM reasoning, uncertainty estimation, and operational routing** into one measurable workflow.

The main technical components are:

```text
Machine Learning
    │
    ├── Classification
    ├── Uncertainty estimation
    └── Conformal routing
             │
             ▼
Retrieval
    │
    └── Milvus + embeddings
             │
             ▼
LLM Engineering
    │
    ├── Local inference
    ├── LangGraph
    └── Grounded reasoning
             │
             ▼
Security Engineering
    │
    ├── Threat intelligence
    ├── Document/OCR processing
    └── Audit telemetry
             │
             ▼
Evaluation
    │
    ├── Classification metrics
    ├── Calibration
    ├── Retrieval diagnostics
    └── Routing metrics
```

---

## ✨ Key Features

| Feature                         | Description                                                                    | Technology                           |
| :------------------------------ | :----------------------------------------------------------------------------- | :----------------------------------- |
| **Agentic Threat Triage**       | Stateful threat-analysis workflow producing structured decisions               | LangGraph                            |
| **Local LLM Reasoning**         | Runs the reasoning layer locally without requiring a hosted LLM API            | Ollama / Llama 3                     |
| **Milvus RAG**                  | Retrieves relevant threat indicators, historical patterns, and SOP context     | Milvus + BGE embeddings              |
| **Conformal Fast Path**         | Bypasses the expensive LLM path for sufficiently confident predictions         | Conformal risk routing               |
| **Live Threat Intelligence**    | Synchronizes current phishing and malicious-URL intelligence before evaluation | OpenPhish, Abuse.ch URLhaus          |
| **PDF & OCR Processing**        | Extracts text from native and scanned PDF documents                            | PyMuPDF, pdfplumber, pypdf, RapidOCR |
| **Domain Adapter Architecture** | Separates domain-specific data loading from the core decision graph            | `DomainAdapter`                      |
| **Zero-Leakage Evaluation**     | Uses grouped stratification to reduce template memorization                    | StratifiedGroupKFold                 |
| **Telemetry & Audit Logging**   | Records prediction, uncertainty, retrieval, routing, and operational metrics   | CSV / logs                           |
| **Interactive Dashboard**       | Allows interactive analysis of text, files, and PDFs                           | Streamlit                            |

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────────────┐
                         │   Live Threat Intelligence   │
                         │                              │
                         │ OpenPhish / Abuse.ch URLhaus│
                         └──────────────┬───────────────┘
                                        │
                                        ▼
┌──────────────────────┐       ┌────────────────────────┐
│ Raw Threat Input     │       │      Milvus RAG        │
│                      │       │                        │
│ • Text               │       │ • Threat indicators   │
│ • Text files         │       │ • Historical patterns  │
│ • PDF documents      │       │ • SOP context          │
└──────────┬───────────┘       └────────────┬───────────┘
           │                                │
           ▼                                │
┌──────────────────────┐                    │
│ Document Extraction  │                    │
│                      │                    │
│ PyMuPDF              │                    │
│ pdfplumber           │                    │
│ pypdf                │                    │
│ RapidOCR             │                    │
└──────────┬───────────┘                    │
           │                                │
           └──────────────┬─────────────────┘
                          ▼
                 ┌──────────────────┐
                 │   ThreatSignal   │
                 │ Normalized input │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Conformal Router │
                 └───────┬─────┬────┘
                         │     │
                confident│     │uncertain
                         │     │
                         ▼     ▼
                    ┌──────┐ ┌──────────────┐
                    │Fast  │ │  Milvus RAG  │
                    │Path  │ │  Retrieval   │
                    └──┬───┘ └──────┬───────┘
                       │             │
                       │             ▼
                       │      ┌──────────────┐
                       │      │  Local LLM   │
                       │      │  Ollama      │
                       │      └──────┬───────┘
                       │             │
                       └──────┬──────┘
                              ▼
                    ┌────────────────────┐
                    │ ThreatResponderGraph│
                    │      LangGraph      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   DecisionOutput    │
                    │                    │
                    │ PHISHING / SPAM    │
                    │ VALID / QUARANTINE │
                    └─────────┬──────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             ┌──────────────┐    ┌───────────────┐
             │ Audit Logs   │    │ Evaluation    │
             │              │    │ Telemetry     │
             └──────────────┘    └───────────────┘
```

The current implementation normalizes inputs into a `ThreatSignal`, enriches them through the RAG layer, processes them with `ThreatResponderGraph`, and records the resulting decision and evaluation telemetry.

---

## Project Structure

```text
universal-threat-responder/
├── adapters/
│   ├── __init__.py
│   ├── phishfuzzer_loader.py            # Zero-leakage dataset loader & TF-IDF vectorization
│   └── rag_spam_adapter.py              # Milvus Vector RAG Adapter
├── data/
│   ├── milvus_threat_intel.db/          # Persistent Milvus Lite vector database
│   └── spam/                            # Evaluation benchmark datasets
├── evals/
│   ├── __init__.py
│   ├── run_evals.py                     # Main evaluation runner with threat-intel sync
│   └── threat_evaluator.py              # Uncertainty & evaluation metrics
├── scripts/
│   ├── update_readme_results.py         # Refreshes benchmark_results.md from latest eval CSVs
│   ├── train_ml_model.py                # Trains and saves the ML classifier used by the dashboard (models/)
│   ├── ingest_threat_intel.py           # Live threat intelligence ingestion (rebuilds `phishing_threat_intel`)
│   └── populate_phishing_vectorstore.py # Indexes PhishFuzzer training emails into the separate `phishfuzzer_training` collection
├── src/
│   ├── __init__.py
│   ├── core.py                          # ThreatResponderGraph & SecurityTools
│   ├── domain_interface.py              # ThreatSignal, DecisionOutput, DomainAdapter
│   └── input_parsers.py                  # JSON and CSV batch input parsers
├── tests/
│   ├── test_core.py                     # Decision routing, enforcement, and audit behavior
│   ├── test_evaluator.py                 # Uncertainty and hallucination telemetry
│   ├── test_high_risk_policy_guard.py    # High-risk VALID override protection
│   ├── test_input_parsers.py             # JSON and CSV input parsing
│   └── test_rag_adapter.py               # Retrieval context and entropy calculations
├── app.py                               # Streamlit dashboard entry point
├── benchmark_results.md                 # Benchmark & telemetry results
├── config.py                            # Configuration & conformal bounds
├── requirements.txt                     # Dependencies
└── README.md                            # Project documentation
```

---

## Interactive Dashboard & Execution Example

The Streamlit prototyping dashboard (`app.py`) allows you to analyze threat signals interactively using raw text, text files, or uploaded PDF documents.

### Dashboard UI Preview

<p align="center">
  <img src="example_screenshots/example_response_page_1.png" width="48%" alt="Dashboard Preview Page 1">
  <img src="example_screenshots/example_response_page_2.png" width="48%" alt="Dashboard Preview Page 2">
</p>

### Sample PDF Triage Execution

When processing an uploaded threat report or email PDF (`2.pdf`), the multi-engine parser and OCR pipeline execute sequentially:

- **OCR Extraction Pipeline**: If no digital text streams are found, the engine initiates optical character recognition.
- **Decision Action**: **QUARANTINE**
- **Confidence & Max Risk Score**: `0.96`
- **Execution Latency**: `0.07 ms`

---

## Benchmark & Telemetry Results

Evaluated on 3,960 PhishFuzzer samples, enabling fast-path routing raised accuracy on resolved rows from 91.15% to 94.47% (macro F1 0.843 → 0.937) and cut the quarantine rate from 58.91% to 42.95%. Compared with the previous benchmark, accuracy on resolved rows and F1 improved and threats wrongly labelled `VALID` dropped sharply (176 → 25 with fast-path enabled), but the system now abstains more often (quarantine rate 33.74% → 42.95%), so the share of all samples answered correctly fell from 59.65% to 53.89%. Quarantined samples count as abstentions and are excluded from accuracy.

See **[benchmark_results.md](benchmark_results.md)** for the full performance summary, confusion matrices, telemetry, and data provenance.

---

# 🗺️ Roadmap

### Evaluation

- [ ] Formalize reproducible dependency versions
- [ ] Add automated CI regression evaluation
- [ ] Expand ablation studies
- [ ] Further calibrate conformal thresholds
- [ ] Evaluate routing quality independently from classification quality

### Retrieval

- [ ] Add sender-domain telemetry
- [ ] Add SPF/DKIM signals
- [ ] Add URL link-entropy features
- [ ] Expand the SOP knowledge base

### Security domains

- [ ] Network intrusion / SIEM adapter
- [ ] NetFlow analysis adapter
- [ ] Transaction-fraud adapter
- [ ] MITRE ATT&CK mapping

### Human oversight

- [ ] Uncertainty-based analyst escalation
- [ ] Human-in-the-loop review queue
- [ ] Analyst feedback collection
- [ ] Review and override telemetry

---

## Setup & Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Llama 3.2

```bash
ollama pull llama3.2
```

Keep Ollama running while using the evaluator or dashboard. If Ollama is not
already running as a background service, open a separate terminal and start it:

```bash
ollama serve
```

Leave that terminal open. Run the evaluation or dashboard commands below from
another terminal.

### Clone & Set Up Environment

```bash
git clone https://github.com/Aylin1/universal-threat-responder.git
cd universal-threat-responder

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows Git Bash
python -m venv .venv
source .venv/Scripts/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest
```

### 1. Run Evaluation — Fast-Path Enabled

```bash
python -m evals.run_evals
```

### 2. Run Evaluation — Full LLM/RAG Path

```bash
python -m evals.run_evals --disable-fast-path
```

Both commands evaluate the PhishFuzzer dataset and write a timestamped telemetry report to:

```text
evals/live_audit_report_results_<fastpath-enabled|fastpath-disabled>_<run_id>.csv
```

After running one or both modes, refresh the Performance Summary in [benchmark_results.md](benchmark_results.md) from the latest report of each mode:

```bash
python -m scripts.update_readme_results
```

### 3. Train the Classifier and Launch the App

The dashboard loads a saved ML classifier from `models/threat_classifier.joblib`. The file is not shipped in the repository (the PhishFuzzer dataset has no published licence, so redistributing a derived model is unconfirmed), so train it once locally (about a minute, no GPU needed):

```bash
python -m scripts.train_ml_model
streamlit run app.py
```

The same model is used with the fast path on or off. The dashboard then follows the same routing as the benchmark: ML probability ≥ 0.95 takes the fast path, otherwise the LLM decides with the ML prediction as context.

Security note: joblib files are pickles and can execute code when loaded. Only load model files you trained yourself.

If the file is missing or fails to load (for example after a scikit-learn/xgboost version change), the dashboard still runs but falls back to retrieval scores only, so most emails with a high-scoring match are quarantined without LLM review. Retraining fixes it.

Please cite the PhishFuzzer paper ("The Phish, The Spam, and The Valid: Generating Feature-Rich Emails for Benchmarking LLMs") when using results from this project.

---

## Notes

This project is a **research/engineering prototype** for experimenting with retrieval-grounded threat triage, conformal routing, local LLMs, and auditable security workflows. Benchmark results are dataset- and configuration-specific and should not be interpreted as production security performance.
