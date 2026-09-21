from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime, timezone


@dataclass
class ThreatSignal:
    """Domain-agnostic representation of a threat event."""
    signal_id: str
    risk_score: float
    confidence: float
    context: Dict[str, Any]
    domain: str = "spam_phishing"
    raw_payload: Any = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class DecisionOutput:
    """Standardized agent decision response."""
    action: str
    reasoning: str
    audit_id: str
    latency_ms: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DomainAdapter(ABC):
    """Base interface contract for domain-specific security adapters."""

    @property
    @abstractmethod
    def domain_name(self) -> str:
        pass

    @property
    @abstractmethod
    def domain_description(self) -> str:
        pass

    @abstractmethod
    def extract_features(self, raw: Any) -> Any:
        pass

    @abstractmethod
    def build_context(self, raw: Any, metadata: Dict = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def validate_input(self, raw: Any) -> bool:
        pass

    def get_prompt_template(self) -> Optional[Any]:
        return None
