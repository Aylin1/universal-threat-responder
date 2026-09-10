import os
from typing import Dict, Any
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.domain_interface import DomainAdapter, ThreatSignal

class RAGSpamAdapter(DomainAdapter):

    def __init__(self, vector_db_path: str = "data/chroma_db"):
        self._domain_name = "spam_phishing"
        self._domain_description = "Vector-backed RAG adapter for email spam, phishing, and scam detection."
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        os.makedirs(vector_db_path, exist_ok=True)
        self.vector_store = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embeddings
        )

    @property
    def domain_name(self) -> str:
        return self._domain_name

    @property
    def domain_description(self) -> str:
        return self._domain_description

    def validate_input(self, raw: Any) -> bool:
        return isinstance(raw, str) and len(raw.strip()) > 0

    import re

    def extract_features(self, raw: Any) -> Dict[str, Any]:
        raw_str = str(raw)
        raw_lower = raw_str.lower()
        
        # Catch links or domain references
        has_urls = any(indicator in raw_lower for indicator in ["http://", "https://", "www.", ".com/", "click here"])

        # Broader spam indicators
        spam_keywords = [
            "urgent", "immediate", "free", "click here", "buy now", "discount",
            "$", "100%", "guaranteed", "unsubscribe", "winner", "prize",
            "lottery", "verify", "account", "limited time", "offer", "cheap",
            "prescription", "viagra", "pills", "mortgage", "refinance", "investment"
        ]
        
        has_urgency = any(w in raw_lower for w in spam_keywords)
        
        return {
            "text_length": len(raw_str),
            "has_urls": has_urls,
            "urgency_language": has_urgency
        }

    def build_context(self, raw_input: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Queries local ChromaDB vector store for relevant policies."""
        try:
            docs = self.vector_store.similarity_search(str(raw_input), k=2)
            retrieved_policies = [doc.page_content for doc in docs]
            if not retrieved_policies:
                retrieved_policies = ["SOP-100: Block unsolicited spam, promotions, phishing, and scam emails. Allow legitimate business email."]
        except Exception:
            retrieved_policies = ["SOP-100: Block unsolicited spam, promotions, phishing, and scam emails. Allow legitimate business email."]

        metadata = metadata or {}
        features = self.extract_features(raw_input)

        return {
            "retrieved_policies": retrieved_policies,
            "raw_input": str(raw_input),
            "text_length": features["text_length"],
            "has_urls": features["has_urls"],
            "urgency_language": features["urgency_language"],
            "metadata": metadata
        }

    def get_prompt_template(self):
        def prompt(signal: ThreatSignal, adapter: DomainAdapter) -> str:
            policies = "\n".join(
                f"  - {p}"
                for p in signal.context.get('retrieved_policies', [])
            )
            email_text = signal.context.get('raw_input', '[No email text available]')
            risk = getattr(signal, "risk_score", 0.0)
            has_urls = signal.context.get("has_urls", False)
            spam_words = signal.context.get("urgency_language", False)

            return f"""SYSTEM: You are an automated security daemon. You must output ONLY raw JSON. Do not write any conversational text, introductions, or conclusions outside the JSON.

SECURITY POLICIES:
{policies}

METRICS:
- Risk Score: {risk}/1.0
- External Links: {has_urls}
- Suspicious Keywords: {spam_words}

EMAIL CONTENT:
\"\"\"
{email_text}
\"\"\"

CLASSIFICATION RULES:
1. Output "BLOCK" if the email appears to be unsolicited marketing, promotional offers, financial scams, external phishing, or generic mass emails.
2. Output "ALLOW" ONLY if the email is legitimate internal company business, trade correspondence, scheduling, personal workplace emails, or technical reports.
3. If suspicious indicators (urgency language, external links, promotional phrasing) are present and the email lacks internal corporate context, default to "BLOCK".

YOU MUST RESPOND WITH ONLY THIS EXACT JSON FORMAT AND NOTHING ELSE:
{{
  "action": "BLOCK" or "ALLOW",
  "reasoning": "Brief 1-sentence explanation"
}}
"""
        return prompt