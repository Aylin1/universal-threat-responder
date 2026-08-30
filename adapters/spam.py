from src.domain_interface import DomainAdapter, ThreatSignal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import re
import pandas as pd

class SpamAdapter(DomainAdapter):
    """Email spam detection adapter."""
    
    @property
    def domain_name(self) -> str:
        return "email_spam"
    
    @property
    def domain_description(self) -> str:
        return "email spam detection and filtering"
    
    def extract_features(self, raw: str):
        """Extract text + structural features from email."""
        if not isinstance(raw, str) or len(raw) < 10:
            raise ValueError("Email text must be at least 10 characters")
        
        # Text features
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        X_text = tfidf.fit_transform([raw])
        
        # Structural features
        def extract_struct(text):
            return {
                'exclamation_count': text.count('!'),
                'url_count': len(re.findall(r'http[s]?://', text)),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'dollar_sign_count': text.count('$'),
                'urgency_score': sum(1 for w in ['urgent', 'immediate', 'act now'] if w in text.lower())
            }
        
        X_struct = StandardScaler().fit_transform([[extract_struct(raw)[k] for k in extract_struct(raw).keys()]])
        return X_text, X_struct, tfidf
    
    def build_context(self, raw: str, metadata: dict = None) -> dict:
        metadata = metadata or {}
        return {
            'sender_domain': metadata.get('sender_domain', 'unknown'),
            'message_age_hours': metadata.get('message_age_hours', 0),
            'has_urls': 'http' in raw,
            'contains_money_mention': '$' in raw,
            'urgency_language': any(w in raw.lower() for w in ['urgent', 'now', 'immediate']),
            'text_length': len(raw)
        }
    
    def validate_input(self, raw: str) -> bool:
        if not isinstance(raw, str):
            raise TypeError("Email must be string")
        if len(raw) < 10:
            raise ValueError("Email too short")
        if len(raw) > 100000:
            raise ValueError("Email too long (max 100KB)")
        return True
    
    def get_prompt_template(self):
        """Specialized prompt for email context."""
        def prompt(signal, adapter):
            ctx_lines = "\n".join(f"  - {k}: {v}" for k, v in signal.context.items()[:6])
            return f"""
You are an autonomous email security agent protecting inboxes from spam.

THREAT ANALYSIS:
- Spam Probability: {signal.risk_score:.2%}
- Model Confidence: {signal.confidence:.2f}
- Sender Domain: {signal.context.get('sender_domain', 'unknown')}
- Has URLs: {signal.context.get('has_urls', False)}
- Contains Money References: {signal.context.get('contains_money_mention', False)}
- Urgency Language Detected: {signal.context.get('urgency_language', False)}
- Message Length: {signal.context.get('text_length', 0)} chars

Actions:
1. QUARANTINE - High-confidence spam, isolate from inbox
2. FLAG REVIEW - Borderline case, human analyst investigation
3. DELIVER - Legitimate email, permit inbox placement
4. RETRAIN - Spam patterns shifting, update model

Respond with: ACTION: [choice] + REASONING
"""
        return prompt