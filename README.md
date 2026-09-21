# 🛡️ Universal Threat Responder

An advanced, domain-agnostic threat triage and decision-engine framework powered by **LangGraph**, **Ollama (Local LLM)**, **Milvus Vector RAG**, **Multi-Engine OCR Extraction**, **Conformal Risk Prediction**, and an interactive **Streamlit Telemetry Dashboard**.

The system normalizes threat signals across security domains — combining phishing detection, recruitment fraud, BEC impersonation, and live malware payload distributions — enriches them with vector-retrieved standard operating procedures (SOPs) and historical indicators, executes stateful decision graphs with optional high-confidence fast-paths, and evaluates agent reasoning with zero target leakage.

---

## Table of Contents

1. [Key Features](#key-features)
2. [System Architecture](#system-architecture)
3. [Project Structure](#project-structure)
4. [Setup & Quickstart](#setup--quickstart)
5. [Benchmark & Telemetry Results](#benchmark--telemetry-results)
6. [Extending to a New Domain](#extending-to-a-new-domain)
7. [Roadmap](#roadmap)

---

## Key Features

| Feature                               | What it does                                                                                                                  | Powered by                                            |
| :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| **Interactive Prototyping Dashboard** | Analyzes text, text files, and PDF uploads interactively with real-time risk scores, latency metrics, and audit traces        | Streamlit (`app.py`)                                  |
| **Multi-Engine PDF & OCR Extraction** | Sequentially parses native digital PDFs, vector text streams, and scanned document images without stream pointer losses       | `PyMuPDF`, `pdfplumber`, `pypdf`, `RapidOCR`          |
| **Live Threat Feed Ingestion**        | Fetches live OpenPhish links, Abuse.ch URLhaus payloads, and behavioral threat patterns at runtime                            | OpenPhish, Abuse.ch URLhaus, Milvus                   |
| **Conformal Fast-Path Routing**       | Automatically bypasses expensive LLM evaluation for high-confidence predictions ($P \le$ `low_bound` or $P \ge$ `high_bound`) |
| **Milvus Vector RAG Retrieval**       | Dynamically indexes and queries historical phishing payloads, indicators, and SOPs using `BAAI/bge-small-en-v1.5` embeddings  | Milvus DB + `HuggingFaceEmbeddings`, `RAGSpamAdapter` |
| **Grounded LLM Decision Graph**       | Stateful decision routing with strictly grounded system prompts that prevent hallucinated threat indicators                   | LangGraph, `ChatOllama`, `SecurityTools`              |
| **Zero-Leakage Dataset Partitioning** | Prevents template memorization via group-stratified cross-validation (`StratifiedGroupKFold`)                                 | `PhishFuzzerDatasetAdapter`                           |

---

## System Architecture

```text
                       +-----------------------------------+
                       | External Live Threat Feeds        |
                       | (OpenPhish, Abuse.ch URLhaus)     |
                       +-----------------+-----------------+
                                         |
                                         v
+--------------------------+     +---------------+
| Raw Text / PDF / File    |     | Milvus DB     |
| (PyMuPDF / RapidOCR)     |     | (Threat Intel)|
+------------+-------------+     +-------+-------+
             |                           |
             v                           |
+--------------------------+             |
| PhishFuzzerDatasetAdapter|             |
|  (StratifiedGroupKFold)  |             |
+------------+-------------+             |
             |                           |
             +-------------+-------------+
                           |
                           v
              +--------------------------+
              |      RAGSpamAdapter      |
              | (Metadata Flattening)    |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |       ThreatSignal       |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |   ThreatResponderGraph   | ---> [Fast-Path Bypass?]
              | (Grounded LangGraph Prompt)      (Optional, if P outside bounds)
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |      DecisionOutput      |
              |  (BLOCK / ALLOW / FLAG)  |
              +------------+-------------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+---------------------+           +---------------------+
|    SecurityTools    |           | DetailedThreatEval  |
|   (Audit Logging)   |           | (Streamlit / Logs)  |
+---------------------+           +---------------------+

Data flows left-to-right/top-to-bottom through the diagram: a domain adapter loads and normalizes raw input into a `ThreatSignal`, the RAG adapter enriches it with retrieved SOPs, the LangGraph-based `ThreatResponderGraph` reasons over the enriched signal to produce a `DecisionOutput`, and that decision is simultaneously logged by `SecurityTools` and independently scored by `LLMJudgeEvaluator`.
```

## Project Structure

```text
universal-threat-responder/
├── adapters/
│   ├── phishfuzzer_loader.py            # Zero-leakage dataset loader & TF-IDF vectorization
│   └── rag_spam_adapter.py              # Milvus Vector RAG Adapter with metadata flattening
├── data/
│   ├── milvus_threat_intel.db           # Persistent Milvus vector database file
│   └── spam/                            # Raw evaluation benchmark datasets
├── evals/
│   ├── run_enron_evals.py               # Main evaluation runner with auto-sync ingestion
│   └── threat_evaluator.py              # Comprehensive uncertainty & hallucination evaluator
├── scripts/
│   ├── ingest_threat_intel.py           # Script to fetch live NVD, KEV, and EPSS feeds
│   ├── seed_threat_intel.py             # Ingests recruitment fraud & SOP-104 signatures
│   └── populate_phishing_vectorstore.py # Indexes training threat patterns into Milvus
├── src/
│   ├── core.py                          # Grounded ThreatResponderGraph & SecurityTools graph
│   └── domain_interface.py              # ThreatSignal, DecisionOutput, DomainAdapter contracts
├── app.py                               # Interactive Streamlit prototyping dashboard
├── config.py                            # Centralized hyperparameters & conformal bounds
├── requirements.txt                     # Dependency specifications
└── README.md                            # Project documentation
```

## Interactive Dashboard & Execution Example

The Streamlit prototyping dashboard (`app.py`) allows you to analyze threat signals interactively using raw text, text files, or uploaded PDF documents.

### Dashboard UI Preview

<p align="center">
  <img src="example_screenshots/example_response_page_1.png" width="48%" alt="Dashboard Preview Page 1">
  <img src="example_screenshots/example_response_page_2.png" width="48%" alt="Dashboard Preview Page 2">
</p>

### Sample PDF Triage Execution

When processing an uploaded threat report or email PDF (`2.pdf`), the multi-engine parser and OCR pipeline execute sequentially:

- **OCR Extraction Pipeline**: If no digital text streams are found, the engine initiates optical character recognition, reporting: _"OCR Engine successfully extracted text from visual PDF layers!"_
- **Decision Action**: **QUARANTINE**.
- **Confidence & Max Risk Score**: `0.96` (96.00%).
- **Execution Latency**: `0.07 ms.

---

## Benchmark & Telemetry Results

The framework was evaluated on an audit dataset of **990 threat signals** processed through both the high-confidence conformal fast-path and the escalated RAG-backed LLM workflow (`live_audit_report_results.csv`).

### Data Provenance & External Sources

- **[PhishFuzzer Dataset (`PhishFuzzer_emails_entity_rephrased_v1.json`)](https://github.com/DataPhish/PhishFuzzer)**: Derived from the **PhishFuzzer** benchmarking corpus ([arXiv:2511.21448](https://arxiv.org/abs/2511.21448)), which provides synthetic and rephrased phishing, spam, and legitimate enterprise email communications designed for robust security model evaluation.

* **OpenPhish Feed (`https://openphish.com/feed.txt`)**: Sourced from **OpenPhish**, a community-driven and automated intelligence feed tracking active, zero-day phishing URLs and credential-harvesting pages.
* **Abuse.ch URLhaus Feed (`https://urlhaus.abuse.ch/downloads/csv_recent/`)**: Sourced from **Abuse.ch URLhaus**, a project dedicated to sharing malicious URLs associated with malware distribution, payload hosting, and botnet command-and-control infrastructure.
* **Behavioral Threat Pattern Registry**: Curated from standard security operating procedures (SOPs) and real-world telemetry patterns covering Business Email Compromise (BEC), executive impersonation, and recruitment/task scams.

### Performance Summary

| Metric / Dimension       | Overall Result | Fast-Path Bypass Route | Escalation RAG LLM Route |
| :----------------------- | :------------: | :--------------------: | :----------------------: |
| **Sample Count & Share** | **990 (100%)** |      396 (40.0%)       |       594 (60.0%)        |
| **Accuracy**             |   **74.24%**   |       **99.75%**       |        **57.24%**        |
| **Macro F1-Score**       |    **0.74**    |        **1.00**        |         **0.51**         |
| **Quarantined Rate**     |   **0.00%**    |         0.00%          |          0.00%           |

### Per-Class Classification Metrics

| Threat Class         | Precision  |   Recall   |  F1-Score  | Support (Samples) |
| :------------------- | :--------: | :--------: | :--------: | :---------------: |
| **PHISHING**         |   0.9720   |   0.8201   |   0.8896   |        339        |
| **SPAM**             |   0.9430   |   0.4474   |   0.6069   |        333        |
| **VALID**            |   0.5641   |   0.9686   |   0.7130   |        318        |
| **Macro Average**    | **0.8264** | **0.7454** | **0.7365** |      **990**      |
| **Weighted Average** | **0.8312** | **0.7424** | **0.7378** |      **990**      |

### Confusion Matrix (True vs. Predicted Agent Actions)

| True Label \ Agent Action | PHISHING |  SPAM   |  VALID  | Total True |
| :------------------------ | :------: | :-----: | :-----: | :--------: |
| **PHISHING**              | **278**  |    3    |   58    |  **339**   |
| **SPAM**                  |    4     | **149** |   180   |  **333**   |
| **VALID**                 |    4     |    6    | **308** |  **318**   |
| **Total Predicted**       | **286**  | **158** | **546** |  **990**   |

> The conformal fast-path successfully intercepted 40% of samples with near-perfect accuracy ($99.75\%$), while the RAG-LLM escalation path handled ambiguous items but experienced lower accuracy ($57.24\%$) due to classification overlap between spam and valid correspondence. Despite these escalation friction points, the hybrid engine maintained a zero structural hallucination rate and robust phishing precision ($97.20\%$).

### Telemetry & Uncertainty Diagnostics

| Metric                         |     Value     |
| :----------------------------- | :-----------: |
| **Mean ML Shannon Entropy**    | `0.4372 bits` |
| **Mean RAG Top-1 Similarity**  |   `0.2297`    |
| **Mean LLM Calibration Error** |   `0.0479`    |

> - Mean ML Shannon Entropy (0.4372 bits): Measures linguistic and URL unpredictability, indicating moderate payload obfuscation across evaluated samples.
> - Mean RAG Top-1 Similarity (0.2297): Quantifies vector alignment with top retrieved threat templates; low similarity reflects novel or zero-day variants requiring dynamic LLM escalation rather than exact matching.
> - Mean LLM Calibration Error (0.0479): Evaluates the gap between model confidence and actual accuracy; a low $\sim 4.79\%$ error proves high model reliability.

---

## Roadmap

- [ ] Formalize stable dependency specifications in `requirements.txt` and containerize via Docker
- [ ] Refine prompt & few-shot discriminator examples in `rag_spam_adapter.py` to separate low-intent spam from valid enterprise email
- [ ] Enrich RAG context vectors with sender domain telemetry, SPF/DKIM authentication flags, and URL link-entropy scores

  > **Implementation Guide**: Because the RAG context must ground the LLM escalation path with technical indicators beyond text body semantics, enriching these vectors involves:
  >
  > 1. Implement a `DomainAdapter` that loads your raw data and normalizes it into a `ThreatSignal`.
  > 2. Pointing `RAGSpamAdapter` at a Milvus collection containing domain-specific SOPs or threat indicators.
  > 3. Reusing `ThreatResponderGraph`' and `DetailedThreatEvaluator` without altering the core decision graph.

- [ ] Calibrate uncertainty thresholds to dynamically route ambiguous borderline samples to human review instead of forced LLM classification
- [ ] Implement automated CI/CD regression testing via GitHub Actions for evaluation pipelines (`run_enron_evals.py`)
- [ ] Add new domain adapters for network intrusion telemetry (SIEM/NetFlow) and transaction fraud analysis
- [ ] Expand the SOP knowledge base and map threat detections directly to MITRE ATT&CK tactics and techniques (TTPs)
- [ ] Build an interactive Human-in-the-Loop (HITL) analyst review queue in the Streamlit dashboard for uncertain samples

## Setup & Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- The Llama 3 model, pulled via:

```bash
ollama pull llama3
```

### Clone & Set Up Environment

```bash
git clone https://github.com/your-username/universal-threat-responder.git
cd universal-threat-responder

# Linux / macOS / Git Bash
python3 -m venv .venv
source .venv/bin/activate

# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not yet configured, install the core dependencies directly:

```bash
pip install pandas langchain-ollama langgraph langchain-chroma langchain-huggingface pyspark
```

## 1. Run Evaluation: Conformal Fast-Path Enabled (Default)

To run evaluations with dynamic conformal routing enabled (bypassing high/low confidence bounds):

```bash
python -m evals.run_enron_evals
```

## 2. Run Evaluation: Full LLM Path (Full LLM Path)

To evaluate every single sample through the complete local LLM reasoning graph without fast-path shortcuts:

```bash
python -m evals.run_enron_evals --disable-fast-path
```

This loads the Enron benchmark samples through `EnronDatasetAdapter`, enriches each one via `RAGSpamAdapter`, routes it through `ThreatResponderGraph`, and writes per-sample audit traces to `evals/live_audit_report_results.csv` as well as `logs/eval_audit.log`.

## 3. Launch the Interactive Dashboard

Start the Streamlit dashboard to test text, text files, or complex PDFs interactively:

```bash
streamlit run app.py
```
