import pandas as pd
from typing import List, Dict, Any, Optional
from src.domain_interface import ThreatSignal


class EnronDatasetAdapter:
    """Supports local Pandas DataFrames and PySpark/Databricks distributed processing."""

    def __init__(self,
                 file_path: str = "data\\spam\\enron_spam_data.csv",
                 spark_session: Optional[Any] = None):
        self.file_path = file_path
        self.spark = spark_session

    def load_as_spark_dataframe(self):
        """Loads dataset as a PySpark DataFrame for Databricks pipeline compatibility."""
        if self.spark is None:
            from pyspark.sql import SparkSession
            self.spark = SparkSession.builder \
                .appName("UniversalThreatResponder-Databricks") \
                .getOrCreate()

        df = self.spark.read.csv(self.file_path, header=True, inferSchema=True)
        return df

    def load_as_test_items(self,
                           sample_fraction: float = 0.01,
                           random_sample: bool = True,
                           seed: int = 42) -> List[Dict[str, Any]]:
        """Loads dataset and structures items for evaluation suite."""
        if self.spark:
            spark_df = self.load_as_spark_dataframe()
            pdf = spark_df.sample(withReplacement=False,
                                  fraction=sample_fraction,
                                  seed=seed).toPandas()
        else:
            pdf = pd.read_csv(self.file_path)
            if random_sample and sample_fraction < 1.0:
                pdf = pdf.sample(frac=sample_fraction, random_state=seed)

        # Identify label column automatically across common dataset variants
        label_col = None
        for col in ["Spam/Ham", "spam/ham", "label", "Label", "target", "Class", "spam"]:
            if col in pdf.columns:
                label_col = col
                break

        test_items = []
        for idx, row in pdf.iterrows():
            msg = str(row.get("Message", row.get("text", ""))).strip()
            subj = str(row.get("Subject", "")).strip()
            if msg.lower() == "nan": 
                msg = ""
            if subj.lower() == "nan": 
                subj = ""

            raw_label = str(row.get(label_col, "")).strip().lower() if label_col else ""

            full_text = f"Subject: {subj}\n\n{msg}" if subj else msg
            full_text = full_text[:1200]

            # Map labels strictly: spam / 1 / true -> BLOCK | ham / 0 / false -> ALLOW
            if raw_label in ["spam", "1", "1.0", "true", "yes"]:
                expected_action = "BLOCK"
            else:
                expected_action = "ALLOW"

            def build_signal_fn(ctx: Dict[str, Any],
                                s_id=f"enron_{idx}",
                                exp_act=expected_action) -> ThreatSignal:
                
                has_urls = ctx.get("has_urls", False)
                urgency = ctx.get("urgency_language", False)
                text_len = ctx.get("text_length", 0)
                
                # Balanced heuristic calculation without ground-truth leakage
                risk = 0.10  # Baseline risk
                if has_urls: 
                    risk += 0.40
                if urgency: 
                    risk += 0.35
                if text_len < 100: 
                    risk += 0.10  # Shorter snippets increase suspiciousness score
                
                risk = min(round(risk, 2), 0.95)
                ctx["domain"] = "spam_phishing"
                
                return ThreatSignal(
                    signal_id=s_id,
                    risk_score=risk,
                    confidence=0.90,
                    context=ctx,
                    domain="spam_phishing",
                    raw_payload=full_text
                )

            test_items.append({
                "input": full_text,
                "expected_action": expected_action,
                "signal_builder": build_signal_fn
            })

        return test_items