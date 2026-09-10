# Universal Threat Responder

An extensible, domain-agnostic threat triage and decision-engine framework powered by **LangGraph**, **Ollama (Llama 3)**, **ChromaDB Vector RAG**, and an **LLM-as-a-Judge** evaluation pipeline.

The system normalizes threat signals across security domains such as email phishing, network intrusion, and fraud, enriches them with vector-retrieved standard operating procedures (SOPs), executes stateful decision graphs, and evaluates agent reasoning against security policies with zero target leakage.

## Project Goals

The goal is to provide a reusable foundation for building and evaluating security triage agents across multiple domains.

The project explores how LLM-based security agents can be combined with:

- Retrieval-Augmented Generation
- Stateful agent workflows
- Domain-specific security policies
- Structured decision interfaces
- Automated reasoning audits
- Local LLM inference
- Distributed data processing

## Key Features

- **Domain-Agnostic Abstraction** : Decouples domain logic from agent decision-making using standardized interfaces such as `ThreatSignal` and `DecisionOutput`.
- **Stateful Workflow Management** : Uses **LangGraph** for conditional decision routing, fallback handling, and tool execution through `SecurityTools`.
- **Vector-Backed RAG Retrieval** : Uses **ChromaDB** and Hugging Face `all-MiniLM-L6-v2` embeddings through `RAGSpamAdapter` to retrieve relevant security policies and SOPs.
- **Distributed & Local Data Pipeline** : Supports local Pandas processing and distributed PySpark execution through `EnronDatasetAdapter`.
- **Zero Target-Leakage Evaluation** : Evaluates LLM reasoning using content-based heuristic scoring without exposing ground-truth target indicators to the decision-making process.
- **Automated LLM Audit Engine** : Uses `LLMJudgeEvaluator` to audit decisions across **Faithfulness**, **Policy Correctness**, and **Hallucination Rate**.

## Design Principles

### Domain Agnostic

The core decision engine is separated from domain-specific logic through adapter interfaces.

The framework can be extended to domains such as:

- Email phishing
- Network intrusion
- Fraud detection
- Suspicious transactions
- Account compromise
- Other security triage workflows

without redesigning the core decision graph.

### Stateful Decision Making

- Instead of treating threat classification as a single LLM prompt, the system models the response process as a stateful workflow.
- This enables explicit routing, fallback handling, and tool execution.

### Retrieval-Augmented Decision Making

- Security policies and SOPs are retrieved dynamically from a vector database rather than being hard-coded directly into the decision workflow.
- This allows the policy knowledge base to evolve independently of the core decision engine.

### Separation of Decision and Evaluation

The evaluation pipeline separates **decision generation** from **decision evaluation**.

Ground-truth target indicators are not directly provided to the decision-making agent. Instead, the system evaluates the generated reasoning based on the available threat evidence, retrieved policies, and decision context.

### Evaluation Metrics

| Metric                 | Description                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| **Faithfulness**       | Measures whether the decision is supported by the available evidence and retrieved context. |
| **Policy Correctness** | Evaluates whether the decision follows the relevant security policies and SOPs.             |
| **Hallucination Rate** | Measures unsupported claims or reasoning that are not grounded in the available context.    |

---

## Latest Benchmark & Audit Results

Evaluated on **337 Enron email samples** using `ThreatResponderGraph`, local `Ollama (Llama 3)`, and `LLMJudgeEvaluator`.

| Metric                         |           Score | Significance                                            |
| ------------------------------ | --------------: | ------------------------------------------------------- |
| **Total Evaluated**            | **337 Signals** | Enron samples processed through `EnronDatasetAdapter`   |
| **Dataset Accuracy Match**     |      **52.52%** | Classification alignment under zero-leakage constraints |
| **Average Judge Faithfulness** |  **0.83 / 1.0** | Consistency between reasoning traces and final actions  |
| **Average Judge Correctness**  |  **0.85 / 1.0** | Alignment of reasoning with retrieved security SOPs     |
| **Hallucination Rate**         |       **0.89%** | Frequency of unsupported or fabricated assertions       |

## System Architecture

```text
                       +--------------------------+
                       |    Raw Security Input    |
                       +------------+-------------+
                                    |
                                    v
                       +--------------------------+
                       |   EnronDatasetAdapter    |
                       |    (Pandas / PySpark)    |
                       +------------+-------------+
                                    |
                                    v
                       +--------------------------+
                       |     RAGSpamAdapter       | <----> ChromaDB
                       |                          |       Vector Store
                       +------------+-------------+
                                    |
                                    | Enriches with
                                    | SOPs & features
                                    v
                       +--------------------------+
                       |      ThreatSignal        |
                       +------------+-------------+
                                    |
                                    v
                       +--------------------------+
                       |   ThreatResponderGraph   |
                       |        (LangGraph)       |
                       +------------+-------------+
                                    |
                                    v
                       +--------------------------+
                       |     DecisionOutput       |
                       |  (BLOCK / ALLOW / FLAG)  |
                       +------------+-------------+
                                    |
                   +----------------+----------------+
                   |                                 |
                   v                                 v
          +---------------------+          +---------------------+
          |    SecurityTools    |          |  LLMJudgeEvaluator  |
          |   (Audit Logging)   |          |   (Reasoning Audit) |
          +---------------------+          +---------------------+
```

## Project Structure

```text
universal-threat-responder/
├── adapters/
│   ├── enron_loader.py          # Enron dataset loader adapter (Pandas / PySpark)
│   └── rag_spam_adapter.py      # ChromaDB Vector RAG DomainAdapter
├── data/
│   ├── chroma_db/               # Persistent ChromaDB vector database
│   └── spam/                    # Raw evaluation benchmark datasets
├── evals/
│   ├── llm_judge.py             # LLM-as-a-Judge quantitative evaluation engine
│   ├── run_enron_evals.py       # Live evaluation pipeline runner
│   └── live_audit_report_results.csv # Exported evaluation audit traces
├── logs/
│   └── eval_audit.log           # SecurityTools persistent execution log
├── src/
│   ├── core.py                  # ThreatResponderGraph & SecurityTools state graph
│   └── domain_interface.py      # ThreatSignal, DecisionOutput, DomainAdapter contracts
├── requirements.txt             # Dependency specifications
└── README.md                    # Project documentation
```

## Setup & Quickstart

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally
- Llama 3 model

Pull the model:

```bash
ollama pull llama3
```

---

## Installation

### Clone Repository & Activate Environment

```bash
git clone [https://github.com/your-username/universal-threat-responder.git](https://github.com/your-username/universal-threat-responder.git)
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

If `requirements.txt` is not yet configured, the core dependencies can be installed with:

```bash
pip install pandas langchain-ollama langgraph langchain-chroma langchain-huggingface pyspark
```

---

## Running the Benchmark

Run the live evaluation pipeline:

```bash
python -m evals.run_enron_evals
```
