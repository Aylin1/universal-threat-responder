import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class VectorStoreConfig:
    uri: str = "./data/milvus_threat_intel.db"
    collection_name: str = "phishing_threat_intel"
    phishing_collection: str = "phishing_threat_intel"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    min_similarity_cutoff: float = 0.10


@dataclass
class ThreatFeedsConfig:
    openphish_url: str = "https://openphish.com/feed.txt"
    urlhaus_csv_url: str = "https://urlhaus.abuse.ch/downloads/csv_recent/"


@dataclass
class FastPathConformalConfig:
    use_fast_path: bool = False  # Default enabled matching README benchmark runs
    high_bound: float = 0.95    # Fast-path AUTO-BLOCK threshold (risk_score >= high_bound)
    low_bound: float = 0.05     # Fast-path AUTO-ALLOW threshold (risk_score <= low_bound)


@dataclass
class ThreatRiskConfig:
    high_risk_threshold: float = 0.95


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model_name: str = "llama3"   # Aligned with 'ollama pull llama3'
    temperature: float = 0.0
    keep_alive: str = "5m"
    confidence_threshold: float = 0.50


@dataclass
class DatasetConfig:
    file_path: str = "data/spam/PhishFuzzer_emails_entity_rephrased_v1.json"
    sample_fraction: float = 1.0
    seed: int = 42
    tfidf_max_features: int = 2000


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    audit_log_path: str = "logs/eval_audit.log"
    report_output_path: str = "evals/live_audit_report_results.csv"


@dataclass
class AppConfig:
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    feeds: ThreatFeedsConfig = field(default_factory=ThreatFeedsConfig)
    conformal: FastPathConformalConfig = field(default_factory=FastPathConformalConfig)
    risk: ThreatRiskConfig = field(default_factory=ThreatRiskConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_directories(self):
        """Helper to create necessary directories automatically."""
        os.makedirs(self.paths.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.paths.audit_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.paths.report_output_path), exist_ok=True)


# Global default configuration instance
cfg = AppConfig()