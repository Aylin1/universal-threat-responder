import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_ollama import ChatOllama

from src.core import ThreatResponderGraph, SecurityTools
from adapters.rag_spam_adapter import RAGSpamAdapter
from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter
from evals.threat_evaluator import DetailedThreatEvaluator
from config import cfg

def sync_threat_feeds():
    ingest_script = Path("scripts/ingest_threat_intel.py")
    
    if ingest_script.exists():
        print("[*] Syncing threat intelligence feeds into Milvus...")
        subprocess.run([sys.executable, str(ingest_script)], check=True)
    else:
        print("[!] Warning: scripts/ingest_threat_intel.py not found. Skipping vector store refresh.")

def run_evaluation():
    parser = argparse.ArgumentParser(description="Run Threat Responder Evaluation")
    parser.add_argument("--disable-fast-path", action="store_true", help="Disable ML fast-path bypass")
    parser.add_argument("--high-bound", type=float, help="Upper probability bound for auto-block")
    parser.add_argument("--low-bound", type=float, help="Lower probability bound for auto-allow")
    parser.add_argument("--sample-fraction", type=float, help="Fraction of dataset to test")
    parser.add_argument("--dataset-path", type=str, help="Override JSON/CSV dataset path")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Give every evaluation run its own audit log.
    audit_log_path = Path(cfg.paths.audit_log_path)
    cfg.paths.audit_log_path = str(
        audit_log_path.with_name(
            f"{audit_log_path.stem}_{run_id}{audit_log_path.suffix}"
        )
    )

    if args.disable_fast_path:
        cfg.conformal.use_fast_path = False
    if args.high_bound is not None:
        cfg.conformal.high_bound = args.high_bound
    if args.low_bound is not None:
        cfg.conformal.low_bound = args.low_bound
    if args.sample_fraction is not None:
        cfg.dataset.sample_fraction = args.sample_fraction
    if args.dataset_path:
        cfg.dataset.file_path = args.dataset_path

    cfg.ensure_directories()
    sync_threat_feeds()

    print(f"\nInitializing Agent Graph with Configured Hyperparameters:")
    print(f" - Fast Path Enabled: {cfg.conformal.use_fast_path}")
    print(f" - Conformal Bounds:  [{cfg.conformal.low_bound:.2f}, {cfg.conformal.high_bound:.2f}]")
    print(f" - Vector Database:   {cfg.vector_store.uri}")
    print(f" - Dataset Path:      {cfg.dataset.file_path}\n")

    llm = ChatOllama(
        model=cfg.llm.model_name, 
        temperature=cfg.llm.temperature, 
        keep_alive=cfg.llm.keep_alive
    )
    tools = SecurityTools(audit_log_path=cfg.paths.audit_log_path)

    agent = ThreatResponderGraph(
        llm=llm,
        tools=tools,
        confidence_threshold=cfg.llm.confidence_threshold,
        use_fast_path=cfg.conformal.use_fast_path
    )

    threat_evaluator = DetailedThreatEvaluator(
        min_similarity_cutoff=cfg.vector_store.min_similarity_cutoff,
        high_bound=cfg.conformal.high_bound,
        low_bound=cfg.conformal.low_bound
    )

    collection_target = getattr(cfg.vector_store, "phishing_collection", "phishing_threat_intel")
    adapter = RAGSpamAdapter(
        milvus_uri=cfg.vector_store.uri,
        collection_name=collection_target
    )
    
    dataset_loader = PhishFuzzerDatasetAdapter(file_path=cfg.dataset.file_path)

    test_items = dataset_loader.load_as_test_items(
        sample_fraction=cfg.dataset.sample_fraction,
        seed=cfg.dataset.seed,
        high_bound=cfg.conformal.high_bound,
        low_bound=cfg.conformal.low_bound
    )

    results = []
    print(f"Running evaluation across {len(test_items)} samples with dynamic conformal routing...\n")

    for item in test_items:
        email_text = item["input"]
        true_label = item["expected_action"]

        context = adapter.build_context(
            raw_input=email_text,
            metadata={
                "source": "enron_eval",
                "sample_count": len(test_items),
            }
        )
        signal = item["signal_builder"](context)

        try:
            decision = agent.respond(signal=signal, adapter=adapter)
            agent_action = decision.action.upper()
            agent_reasoning = decision.reasoning
            agent_confidence = getattr(decision, "confidence", 0.0)
            route_taken = getattr(decision, "metadata", {}).get(
                "evaluation_mode", 
                signal.context.get("evaluation_mode", "UNKNOWN")
            )
        except Exception as e:
            agent_action = "QUARANTINE"
            agent_reasoning = f"Graph execution failed: {str(e)}"
            agent_confidence = 0.0
            route_taken = "FALLBACK"

        llm_payload = {
            "action": agent_action,
            "confidence": agent_confidence,
            "reasoning": agent_reasoning
        }
        telemetry = threat_evaluator.evaluate_signal(signal, llm_payload)

        is_exact_match = (agent_action == true_label)
        is_quarantined = (agent_action == "QUARANTINE")

        results.append({
            "signal_id": signal.signal_id,
            "true_label": true_label,
            "agent_action": agent_action,
            "route_taken": route_taken,
            "agent_confidence": agent_confidence,
            "label_match": is_exact_match,
            "is_quarantined": is_quarantined,
            "high_risk_hits": context.get("high_risk_hits", 0),
            "max_risk_score": context.get("max_risk_score", 0.0),

            "ml_risk_prob": telemetry["ml_uncertainty"]["ml_risk_raw_probability"],
            "ml_shannon_entropy": telemetry["ml_uncertainty"]["ml_shannon_entropy_bits"],
            "rag_similarity": telemetry["rag_uncertainty"]["rag_top1_similarity"],
            "structural_hallucination_rate": telemetry["hallucination_metrics"]["structural_hallucination_rate"],
            "calibration_error": telemetry["llm_calibration"]["verbalization_calibration_error"],

            "flag_epistemic_uncertainty": telemetry["vector_flags"]["high_epistemic_uncertainty"],
            "flag_retrieval_uncertainty": telemetry["vector_flags"]["high_retrieval_uncertainty"],
            "flag_structural_hallucination": telemetry["vector_flags"]["structural_hallucination_detected"],
            "flag_overconfident_llm": telemetry["vector_flags"]["overconfident_llm"]
        })

    results_df = pd.DataFrame(results)
    resolved_df = results_df[~results_df['is_quarantined']].copy()

    if not resolved_df.empty:
        y_true = resolved_df['true_label']
        y_pred = resolved_df['agent_action']

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        resolved_accuracy = (y_true == y_pred).mean()
    else:
        precision = recall = f1 = resolved_accuracy = 0.0

    fast_path_rate = (results_df['route_taken'] == 'FAST_PATH_BYPASS').mean() * 100

    print("=" * 60)
    print("📊 LIVE CONFORMAL & PROACTIVE THREAT TELEMETRY REPORT")
    print("=" * 60)
    print(f"Total Evaluated:                    {len(results_df)}")
    print(f"Fast-Path Bypass Rate:              {fast_path_rate:.2f}%")
    print(f"High-Risk Threat Indicator Hits:   {results_df['high_risk_hits'].sum()}")
    print(f"Quarantined Action Rate:            {results_df['is_quarantined'].mean() * 100:.2f}%")
    print("-" * 60)
    print("UNCERTAINTY TELEMETRY METRICS:")
    print(f"Mean ML Shannon Entropy (bits):     {results_df['ml_shannon_entropy'].mean():.4f}")
    print(f"Mean RAG Top-1 Similarity:          {results_df['rag_similarity'].mean():.4f}")
    print(f"Mean LLM Calibration Error:         {results_df['calibration_error'].mean():.4f}")
    print("-" * 60)

    if not resolved_df.empty:
        print("THREAT DETECTION METRICS (Resolved Items Only):")
        print(f"Accuracy:                           {resolved_accuracy * 100:.2f}%")
        print(f"Precision (Macro 3-Class):          {precision * 100:.2f}%")
        print(f"Recall (Macro 3-Class):             {recall * 100:.2f}%")
        print(f"F1-Score (Macro 3-Class):           {f1:.3f}")
    print("=" * 60)

    output_path = Path("evals") / f"live_audit_report_results_{run_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved detailed telemetry log to `{output_path}`")


if __name__ == "__main__":
    run_evaluation()