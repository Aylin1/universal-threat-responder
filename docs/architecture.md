# Architecture & Flow

Diagrams are written in [Mermaid](https://mermaid.js.org/) and render natively on GitHub and in VS Code (Markdown Preview Mermaid support).

## 1. Architectural diagram

```mermaid
flowchart TB
    subgraph EXT["External sources"]
        OP["OpenPhish feed"]
        UH["Abuse.ch URLhaus CSV"]
        PF["PhishFuzzer dataset<br/>data/spam/*.json"]
    end

    subgraph OFFLINE["Offline / setup (scripts/)"]
        ING["ingest_threat_intel.py<br/>live IOCs + SOP / social-eng docs"]
        POP["populate_phishing_vectorstore.py<br/>training docs"]
    end

    subgraph STORE["Storage"]
        MIL[("Milvus Lite<br/>phishing_threat_intel<br/>BGE-small embeddings")]
        AUDIT[("logs/audit.log<br/>JSONL")]
        CSV[("evals/*.csv<br/>benchmark_results.md")]
    end

    subgraph ENTRY["Entry points"]
        APP["app.py<br/>Streamlit dashboard"]
        EVAL["evals/run_evals.py<br/>batch evaluation"]
    end

    subgraph INPUT["Input layer"]
        PARSE["src/input_parsers.py<br/>JSON / CSV / text"]
        PDF["PDF extraction<br/>PyMuPDF → pdfplumber → pypdf → RapidOCR"]
        LOADER["adapters/phishfuzzer_loader.py<br/>TF-IDF + structural features<br/>calibrated ML classifier<br/>StratifiedGroupKFold"]
    end

    subgraph ADAPT["Domain adapter (adapters/rag_spam_adapter.py)"]
        RAG["RAGSpamAdapter<br/>build_context(): top-5 similarity search<br/>max_risk_score, SOP mapping,<br/>prompt template"]
    end

    subgraph CORE["Core (src/core.py)"]
        SIG["ThreatSignal<br/>(src/domain_interface.py)"]
        ROUTER{"route_threat_signal<br/>risk_score ≥ high_bound?"}
        GRAPH["ThreatResponderGraph<br/>(LangGraph StateGraph)"]
        TOOLS["SecurityTools<br/>block / quarantine / allow"]
        OUT["DecisionOutput<br/>action, reasoning, confidence,<br/>audit_id, latency, metadata"]
    end

    LLM["Ollama – llama3<br/>(local LLM)"]
    CFG["config.py<br/>cfg: conformal bounds, LLM,<br/>vector store, paths"]
    EVALM["evals/threat_evaluator.py<br/>uncertainty, calibration,<br/>retrieval metrics"]

    OP --> ING
    UH --> ING
    PF --> POP
    PF --> LOADER
    ING --> MIL
    POP --> MIL
    EVAL -. "sync_threat_feeds()" .-> ING

    APP --> PARSE
    APP --> PDF
    PARSE --> RAG
    PDF --> RAG
    EVAL --> LOADER
    LOADER --> RAG

    RAG <--> MIL
    RAG --> SIG
    LOADER -. "risk_score = ML max prob<br/>ml_predicted_class" .-> SIG
    RAG -. "dashboard: risk_score = max_risk_score" .-> SIG

    SIG --> GRAPH
    GRAPH --- ROUTER
    GRAPH <--> LLM
    GRAPH --> TOOLS
    TOOLS --> AUDIT
    GRAPH --> OUT

    OUT --> APP
    OUT --> EVALM
    EVALM --> CSV

    CFG -.-> APP
    CFG -.-> EVAL
    CFG -.-> GRAPH
    CFG -.-> RAG
```

### Component summary

| Layer | Files | Responsibility |
|---|---|---|
| Entry points | `app.py`, `evals/run_evals.py` | Interactive triage UI; batch evaluation (with optional live threat-feed sync) |
| Input | `src/input_parsers.py`, PDF/OCR in `app.py`, `adapters/phishfuzzer_loader.py` | Turn raw text / files / PDFs / benchmark rows into text payloads |
| Adapter | `adapters/rag_spam_adapter.py`, `src/domain_interface.py` | Domain-specific retrieval, context building, prompt template; `DomainAdapter` contract keeps the core domain-agnostic |
| Core | `src/core.py` | LangGraph workflow, conformal routing, policy guard, audit tooling |
| Knowledge base | `scripts/*.py`, `data/milvus_threat_intel.db` | Live IOCs (OpenPhish, URLhaus), SOPs and phishing patterns embedded with `BAAI/bge-small-en-v1.5` |
| Evaluation | `evals/threat_evaluator.py`, `scripts/update_readme_results.py` | Accuracy, calibration, entropy, retrieval similarity, routing metrics |

## 2. Flow chart – decision workflow

This mirrors the LangGraph built in `ThreatResponderGraph._build_graph()`.

```mermaid
flowchart TD
    A([Raw threat input<br/>text / file / PDF / dataset row]) --> B{PDF?}
    B -- yes --> C["Extract text<br/>PyMuPDF → pdfplumber → pypdf"]
    C --> D{Digital text found?}
    D -- no --> E["RapidOCR on rendered pages"]
    D -- yes --> F
    E --> F
    B -- no --> F["Normalize text payload"]

    F --> G["RAGSpamAdapter.build_context()<br/>Milvus top-5 similarity search<br/>drop matches with cosine &lt; 0.55"]
    G --> H["Build ThreatSignal<br/>risk_score, context, ml_predicted_class"]

    H --> I{"use_fast_path AND<br/>risk_score ≥ high_bound (0.95)?"}

    I -- "yes: confident" --> J["fast_path_bypass<br/>action = ML predicted class<br/>mode = FAST_PATH_BYPASS"]

    I -- "no: uncertain" --> K["build_prompt<br/>adapter template + retrieved context + SOPs"]
    K --> L["evaluate_threat<br/>Ollama llama3 invoke"]
    L --> M{LLM error?}
    M -- yes --> N["fallback_handler<br/>QUARANTINE, confidence 0"]
    M -- no --> O["Parse JSON action<br/>VALID / SPAM / PHISHING / QUARANTINE<br/>(heuristic fallback → QUARANTINE)"]
    O --> P{"action = VALID AND<br/>max_risk ≥ high_bound?"}
    P -- yes --> Q["Policy guard:<br/>override to QUARANTINE"]
    P -- no --> R[Keep LLM action]

    J --> S
    Q --> S
    R --> S["execute_action"]
    S --> T{action}
    T -- PHISHING --> U["tools.block()"]
    T -- SPAM --> V["tools.quarantine()"]
    T -- VALID --> W["tools.allow()"]
    T -- "QUARANTINE / other" --> X["tools.quarantine()"]
    N --> Y

    U --> Y["Write audit entry<br/>logs/audit.log"]
    V --> Y
    W --> Y
    X --> Y
    Y --> Z(["DecisionOutput<br/>action, reasoning, confidence,<br/>audit_id, latency, uncertainty metadata"])
```

## 3. Sequence – one request through the system

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as Streamlit app.py
    participant AD as RAGSpamAdapter
    participant MV as Milvus
    participant G as ThreatResponderGraph
    participant LLM as Ollama llama3
    participant T as SecurityTools

    User->>UI: text / file / PDF
    UI->>AD: build_context(text)
    AD->>MV: similarity_search_with_score(k=5)
    MV-->>AD: matches + distances
    AD-->>UI: context (max_risk_score, SOPs, matches)
    UI->>G: respond(ThreatSignal, adapter)
    alt risk_score ≥ high_bound
        G->>G: fast_path_bypass
    else escalate
        G->>AD: get_prompt_template()
        G->>LLM: invoke(prompt)
        LLM-->>G: JSON decision
        G->>G: parse + policy guard
    end
    G->>T: block / quarantine / allow
    T-->>G: audit_id (logged to audit.log)
    G-->>UI: DecisionOutput
    UI-->>User: action, confidence, reasoning, telemetry
```

## 4. Evaluation pipeline

```mermaid
flowchart LR
    S["sync_threat_feeds()<br/>ingest_threat_intel.py"] --> L["PhishFuzzerDatasetAdapter<br/>group-aware split,<br/>calibrated classifier"]
    L --> I["test items<br/>input + expected_action + signal_builder"]
    I --> R["RAGSpamAdapter.build_context()"]
    R --> A["ThreatResponderGraph.respond()<br/>fast path on / off"]
    A --> E["DetailedThreatEvaluator<br/>accuracy, F1, calibration error,<br/>entropy, retrieval similarity"]
    E --> C["evals/live_audit_report_*.csv"]
    C --> U["update_readme_results.py"] --> B["benchmark_results.md"]
```
