from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone


@dataclass
class ThreatSignal:
    """
    Domain-agnostic representation of a threat event.
    Each adapter transforms raw input into this format.
    """
    signal_id: str                          # Unique identifier
    risk_score: float                       # 0.0–1.0 from ML model
    confidence: float                       # Model confidence
    context: Dict[str, Any]                 # Domain-specific metadata
    domain: str = "spam_phishing"           # Domain identifier
    raw_payload: Any = None                 # Preserved original input (optional)
    timestamp: Optional[datetime] = None    # Auto-filled UTC timestamp

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class DecisionOutput:
    """Standardized agent response."""
    action: str                             # BLOCK | FLAG | ALLOW | RETRAIN
    reasoning: str                          # Natural language explanation
    audit_id: str                           # Traceable log entry
    latency_ms: float                       # Decision execution time
    metadata: Dict[str, Any] = None         # Additional context

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DomainAdapter(ABC):
    """
    Base class for domain-specific adapters.
    Implement these methods to plug in any new threat domain.
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Human-readable domain identifier (e.g., 'spam_phishing')."""
        pass

    @property
    @abstractmethod
    def domain_description(self) -> str:
        """Brief description for agent system prompts."""
        pass

    @abstractmethod
    def extract_features(self, raw: Any) -> Any:
        """Transform raw input into feature representation."""
        pass

    @abstractmethod
    def build_context(self, raw: Any, metadata: Dict = None) -> Dict[str, Any]:
        """Build domain-specific metadata dictionary for agent context."""
        pass

    @abstractmethod
    def validate_input(self, raw: Any) -> bool:
        """Check if raw input meets minimum requirements."""
        pass

    def get_prompt_template(self) -> Optional[Any]:
        """Override for domain-customized prompt template/function."""
        return None