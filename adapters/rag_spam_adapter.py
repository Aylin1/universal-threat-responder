import os
import re
import math
from typing import Dict, Any, List
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

from src.domain_interface import DomainAdapter, ThreatSignal
from config import cfg

MIN_RELEVANT_SIMILARITY = 0.10

DEFAULT_POLICIES = [
    "SOP-101: ALLOW legitimate corporate correspondence, benign newsletters, and verifiable automated alerts.",
    "SOP-102: QUARANTINE unsolicited bulk commercial email, promotional marketing, or high-volume spam.",
    "SOP-103: BLOCK active phishing links, credential harvesting attempts, or malicious payload URLs.",
    "SOP-104: BLOCK recruitment fraud, task scams, and social engineering vectors (e.g., post-rejection practicum/bootcamp pivots or fake HR onboarding)."
]

def compute_shannon_entropy(p: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return - (p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

class RAGSpamAdapter(DomainAdapter):
    def __init__(
        self, 
        milvus_uri: str = "./data/milvus_threat_intel.db",
        collection_name: str = "phishing_threat_intel"
    ):
        self._domain_name = "spam_phishing"
        self._domain_description = "Milvus-backed RAG threat intelligence adapter for email, phishing, and social engineering."

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )

        os.makedirs(os.path.dirname(milvus_uri), exist_ok=True)
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            collection_name=collection_name,
            connection_args={"uri": milvus_uri}
        )

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def domain_description(self) -> str:
        return self._domain_description

    def validate_input(self, raw: Any) -> bool:
        return isinstance(raw, str) and len(raw.strip()) > 0

    def extract_features(self, raw: Any) -> Any:
        return str(raw)

    def _extract_search_query(self, raw_input: str) -> str:
        urls = re.findall(r'https?://[^\s]+', raw_input)
        
        subject_match = re.search(r'Subject:\s*(.*)', raw_input, re.IGNORECASE)
        subject = subject_match.group(1).strip() if subject_match else ""

        query_components = []
        if subject:
            query_components.append(f"Subject: {subject}")
        if urls:
            query_components.append(" ".join(urls[:3]))
        
        clean_body = re.sub(r'\s+', ' ', raw_input).strip()
        query_components.append(clean_body[:350])

        return " | ".join(query_components)

    def build_context(self, raw_input: str, metadata: dict = None) -> dict:
        search_results = self.vector_store.similarity_search_with_score(
            query=raw_input, 
            k=5
        )

        retrieved_matches = []
        max_risk_score = 0.0
        max_risk_sop = ""
        high_risk_hits = 0
        top_threat_source = "Known Threat Intelligence" 

        for doc, score in search_results:
            doc_meta = doc.metadata or {}
            content = doc.page_content
            
            # Extract Risk Score and dynamic source
            distance = float(score)
            calculated_risk = max(0.0, min(1.0, 1.0 - distance))

            # --- FILTER OUT IRRELEVANT MATCHES ---
            if calculated_risk < MIN_RELEVANT_SIMILARITY:
                continue
            # -------------------------------------
            
            risk = float(doc_meta.get("risk_score", round(calculated_risk, 2)))
            current_source = doc_meta.get("source", "Known Threat Intelligence")
            
            if risk >= cfg.risk.high_risk_threshold:
                high_risk_hits += 1

            # Derive SOP / Category fallback
            sop_id = doc_meta.get("sop_id")
            category = doc_meta.get("category")
            indicator_type = doc_meta.get("indicator_type")

            if not sop_id:
                if "login" in content.lower() or "http" in content.lower():
                    sop_id = "SOP-103"
                    category = "phishing_link"
                    indicator_type = "url_pattern"
                elif "practicum" in content.lower() or "consulting" in content.lower():
                    sop_id = "SOP-104"
                    category = "recruitment_fraud"
                    indicator_type = "social_engineering"
                else:
                    sop_id = "SOP-101"
                    category = "benign_notification"
                    indicator_type = "platform_alert"

            if risk > max_risk_score:
                max_risk_score = risk
                max_risk_sop = sop_id
                top_threat_source = current_source

            preview_content = content.replace("\n", " ")
            if len(preview_content) > 120:
                preview_content = preview_content[:117] + "..."

            retrieved_matches.append({
                "Indicator Signature": preview_content,
                "Type": indicator_type,
                "Risk Score": risk,
                "Mapped SOP": sop_id,
                "Category": category,
                "Source": current_source,
                "Vector Distance": round(distance, 4)
            })

        return {
            "raw_input": raw_input,
            "retrieved_matches": retrieved_matches,
            "max_risk_score": max_risk_score,
            "max_risk_sop": max_risk_sop,
            "high_risk_hits": high_risk_hits,
            "threat_source": top_threat_source,
            "retrieval_confidence": max([max(0.0, 1.0 - m["Vector Distance"]) for m in retrieved_matches], default=0.0),
            "retrieved_policies": DEFAULT_POLICIES
        }
    
    def get_prompt_template(self):
        def prompt(signal: ThreatSignal, adapter: DomainAdapter) -> str:
            ctx = signal.context or {}
            email_text = ctx.get("raw_input", signal.raw_payload or "")
            pred_class = ctx.get("ml_predicted_class", "UNKNOWN")
            confidence = getattr(signal, "risk_score", 0.50)
            max_risk_score = float(ctx.get("max_risk_score", 0.0))
            max_risk_sop = ctx.get("max_risk_sop", "UNKNOWN")

            return f"""You are a senior enterprise security analyst categorizing an incoming email into one of three classes: PHISHING, SPAM, or VALID.

TELEMETRY:
- Local ML Prediction: {pred_class} (Confidence: {confidence:.4f})

OPERATIONAL RULES:
1. PHISHING: Credential harvesting, malicious links, fraud, or manipulative urgency (e.g., "urgent account update").
2. SPAM: Unsolicited marketing, bulk newsletters, or low-value promotions without malicious intent.
3. VALID: Legitimate business correspondence or expected automated alerts.

---
EXAMPLE (PHISHING):
Email: "URGENT: Your IT mailbox is full. Click here to verify your credentials."
Output:
{{
  "action": "PHISHING",
  "confidence": 0.95,
  "reasoning": "Manipulative urgency requesting credential verification via an external link."
}}

EXAMPLE (SPAM):
Email: "Act now to get 50% off our new enterprise SEO tool! Unsubscribe here."
Output:
{{
  "action": "SPAM",
  "confidence": 0.90,
  "reasoning": "Unsolicited promotional bulk email, no immediate security threat."
}}
---

EVALUATE THE FOLLOWING EMAIL:

Email Text:
\"\"\"
{email_text}
\"\"\"

Respond strictly with a raw JSON object containing "action", "confidence", and "reasoning".
"""
        return prompt