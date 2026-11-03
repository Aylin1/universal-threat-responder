from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
    """Base interface contract for domain-specific security adapters.

    The core engine knows nothing about PHISHING/SPAM/VALID. Everything domain specific
    (action vocabulary, what each action does, how raw LLM text maps to an action, and the
    post-LLM policy guard) lives on the adapter. The defaults below are a generic
    ALLOW / BLOCK / QUARANTINE domain.
    """

    # ---- identity -------------------------------------------------------
    @property
    @abstractmethod
    def domain_name(self) -> str:
        pass

    @property
    @abstractmethod
    def domain_description(self) -> str:
        pass

    # ---- data contract --------------------------------------------------
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

    # ---- action vocabulary (override per domain) ------------------------
    @property
    def allowed_actions(self) -> Tuple[str, ...]:
        return ("ALLOW", "BLOCK", "QUARANTINE")

    @property
    def benign_action(self) -> str:
        return "ALLOW"

    @property
    def fallback_action(self) -> str:
        """Safe abstention used on errors, parse failures, low confidence, no majority."""
        return "QUARANTINE"

    @property
    def ioc_block_action(self) -> str:
        """Action taken on a deterministic known-bad indicator match."""
        return "BLOCK"

    @property
    def action_effects(self) -> Dict[str, str]:
        """Maps action -> SecurityTools method name ('block' | 'quarantine' | 'allow')."""
        return {"BLOCK": "block", "QUARANTINE": "quarantine", "ALLOW": "allow"}

    def normalize_action(self, raw_action: str) -> str:
        """Map free-form LLM output to an allowed action, else the fallback action."""
        a = str(raw_action or "").upper().strip()
        return a if a in self.allowed_actions else self.fallback_action

    def apply_policy_guard(self, signal: ThreatSignal, action: str) -> Tuple[str, bool, List[str]]:
        """Optional post-LLM veto. Returns (final_action, applied, reasons). Default: no-op."""
        return action, False, []
