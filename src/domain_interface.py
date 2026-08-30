from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from datetime import datetime

@dataclass
class ThreatSignal:
    """
    Domain-agnostic representation of a threat event.
    Each adapter transforms raw input into this format.
    """
    signal_id: str                           # Unique identifier
    risk_score: float                        # 0.0–1.0 from ML model
    confidence: float                        # Model confidence
    context: Dict[str, Any]                  # Domain-specific metadata
    raw_payload: Any                         # Original input preserved
    timestamp: datetime = None               # Auto-filled if None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class DecisionOutput:
    """Standardized agent response."""
    action: str                              # BLOCK | FLAG | ALLOW | RETRAIN
    reasoning: str                           # Natural language explanation
    audit_id: str                            # Traceable log entry
    latency_ms: float                        # Decision time
    metadata: Dict[str, Any] = None          # Additional context
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DomainAdapter(ABC):
    """
    Base class for domain-specific adapters.
    Implement these three methods to plug in any new domain.
    """
    
    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Human-readable domain identifier (e.g., 'email_spam')."""
        pass
    
    @property
    @abstractmethod
    def domain_description(self) -> str:
        """Brief description for agent prompts."""
        pass
    
    @abstractmethod
    def extract_features(self, raw: Any) -> Any:
        """
        Transform raw input → feature vector for ML model.
        Return: (X_vectorizer_fitted, X_struct_array, vectorizer_object)
        """
        pass
    
    @abstractmethod
    def build_context(self, raw: Any, metadata: Dict = None) -> Dict[str, Any]:
        """
        Build domain-specific metadata dict for agent prompt.
        Return: Dictionary with meaningful keys for decision context.
        """
        pass
    
    @abstractmethod
    def validate_input(self, raw: Any) -> bool:
        """
        Check if input meets minimum quality requirements.
        Raise ValueError if invalid.
        """
        pass
    
    def get_prompt_template(self) -> str:
        """Override for custom agent prompt wording."""
        return None  # Use default universal prompt