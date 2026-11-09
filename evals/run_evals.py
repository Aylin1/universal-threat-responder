import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
from langchain_ollama import ChatOllama

from adapters.phishfuzzer_loader import PhishFuzzerDatasetAdapter
from adapters.rag_spam_adapter import RAGSpamAdapter
from config import cfg
from evals.analyze_results import summarize, to_markdown
from evals.run_metadata import build_run_metadata
from evals.threat_evaluator import DetailedThreatEvaluator
from src.config_access import cfg_get
from src.core import SecurityTools, ThreatResponderGraph

logger = logging.getLogger(__name__)
MANIFEST_PATH = Path("./data/history_manifest.json")


def sync_threat_feeds(refresh: bool):
    """Builds/refreshes the pattern collection and IOC snapshot. It does NOT touch the email-history
    collection (the old version of this step silently wiped it before every run)."""
    script = Path("scripts/ingest_threat_intel.py")
    if not script.exists():
        print("[!] scripts/ingest_threat_intel.py not found; skipping feed sync.")
        return
    cmd = [sys.executable, str(script)] + (["--refresh-feeds"] if refresh else [])
    subprocess.run(cmd, check=True)


def verify_history_index(loader: PhishFuzzerDatasetAdapter):
    if not MANIFEST_PATH.exists():
        raise SystemExit("No data/history_manifest.json: run `python scripts/populate_phishing_vectorstore.py` first "
                         "(the retrieval index of labelled training emails has not been built).")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("split_fingerprint") != loader.split_fingerprint():
        raise SystemExit("The indexed email history was built from a DIFFERENT split (dataset/seed/fold changed). "
                         "Re-run scripts/populate_phishing_vectorstore.py with the same --dataset-path and --fold.")
    print(f"[ok] History index matches split ({manifest['n_docs']} labelled emails indexed).")
    return manifest


def run_evaluation():
    parser = argparse.ArgumentParser(description="Run Threat Responder Evaluation")
    parser.add_argument("--disable-fast-path", action="store_true")
    parser.add_argument("--enable-fast-path", action="store_true")
    parser.add_argument("--legacy-policy", action="store_true",
                        help="run the historical LLM/guard policy instead of mandatory agreement review")
    parser.add_argument("--alpha", type=float, help="conformal miscoverage level (default 0.05)")
    parser.add_argument("--fold", type=int, default=0, help="outer CV fold used as test split")
    parser.add_argument("--sample-fraction", type=float)
    parser.add_argument("--dataset-path", type=str)
    parser.add_argument("--llm-samples", type=int, default=1, help="LLM draws per email for self-consistency")
    parser.add_argument("--refresh-feeds", action="store_true", help="re-download OpenPhish/URLhaus")
    # deprecated, accepted so old command lines keep working
    parser.add_argument("--high-bound", type=float)
    parser.add_argument("--low-bound", type=float)
    args = parser.parse_args()

    if args.enable_fast_path and not args.legacy_policy:
        parser.error("--enable-fast-path requires --legacy-policy; agreement review runs both models")
    if args.enable_fast_path and args.disable_fast_path:
        parser.error("choose either --enable-fast-path or --disable-fast-path")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if args.disable_fast_path:
        cfg.conformal.use_fast_path = False
    if args.enable_fast_path:
        cfg.conformal.use_fast_path = True
    if args.sample_fraction is not None:
        cfg.dataset.sample_fraction = args.sample_fraction
    if args.dataset_path:
        cfg.dataset.file_path = args.dataset_path
    if args.high_bound is not None or args.low_bound is not None:
        print("[!] --high-bound/--low-bound are deprecated: routing uses the conformal prediction set (--alpha).")

    cfg.ensure_directories()
    sync_threat_feeds(args.refresh_feeds)

    print("\nBuilding dataset splits, fitting classifier and calibrating conformal router...")
    loader = PhishFuzzerDatasetAdapter(file_path=cfg.dataset.file_path, alpha=args.alpha, outer_fold=args.fold)
    loader.assert_no_leakage()
    history_manifest = verify_history_index(loader)
    leak = loader.leakage_report()
    print("Leakage report:", json.dumps(leak))
    print("Conformal check on held-out test split:", json.dumps(
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in loader.conformal_report.items()
         if k != "coverage_per_class"}))
    print("  per-class coverage:", {k: round(v, 3) for k, v in loader.conformal_report["coverage_per_class"].items()})

    use_fast = bool(cfg_get("conformal", "use_fast_path", True))
    mode = ("fastpath-enabled" if use_fast else "fastpath-disabled") if args.legacy_policy else "agreement-review"
    stem = f"live_audit_report_results_{mode}_{run_id}"
    run_dir = Path("evals") / "results" / "runs" / f"{mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"\n - Fast path enabled: {use_fast}\n - Conformal alpha:   {loader.alpha}\n"
          f" - Vector DB:         {cfg.vector_store.uri}\n - Dataset:           {cfg.dataset.file_path}\n")

    llm = ChatOllama(model=cfg.llm.model_name, temperature=cfg.llm.temperature, keep_alive=cfg.llm.keep_alive)
    tools = SecurityTools(audit_log_path=str(run_dir / "audit.jsonl"))
    agent = ThreatResponderGraph(llm=llm, tools=tools, uncertainty_samples=args.llm_samples,
                                 confidence_threshold=cfg.llm.confidence_threshold, use_fast_path=use_fast,
                                 require_agreement_review=not args.legacy_policy)
    evaluator = DetailedThreatEvaluator(min_similarity_cutoff=cfg.vector_store.min_similarity_cutoff)
    adapter = RAGSpamAdapter(milvus_uri=cfg.vector_store.uri)

    test_items = loader.load_as_test_items(sample_fraction=cfg.dataset.sample_fraction, seed=cfg.dataset.seed)
    print(f"Running evaluation across {len(test_items)} samples...\n")
    metadata = build_run_metadata(run_id=run_id, mode=mode, args=args, config=cfg, loader=loader,
                                  adapter=adapter, history_manifest=history_manifest, sample_count=len(test_items))
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    results = []
    for item in test_items:
        true_label = item["expected_action"]

        t0 = perf_counter()
        context = adapter.build_context(raw_input=item["input"],
                                        metadata={"source": "phishfuzzer_eval", "sample_count": len(test_items)})
        retrieval_ms = (perf_counter() - t0) * 1000
        signal = item["signal_builder"](context)

        meta = {}
        try:
            t1 = perf_counter()
            decision = agent.respond(signal=signal, adapter=adapter)
            graph_ms = (perf_counter() - t1) * 1000
            action = decision.action.upper()
            reasoning = decision.reasoning
            confidence = decision.confidence if decision.confidence is not None else 0.0
            meta = decision.metadata or {}
            route = meta.get("evaluation_mode", signal.context.get("evaluation_mode", "UNKNOWN"))
        except Exception as exc:
            logger.exception("Graph execution failed for signal %s", signal.signal_id)
            graph_ms = float("nan")
            action, reasoning, confidence, route = "QUARANTINE", f"Graph execution failed: {exc}", 0.0, "FALLBACK"

        tel = evaluator.evaluate_signal(signal, {"action": action, "confidence": confidence, "reasoning": reasoning})
        ctx = signal.context
        probs = ctx.get("ml_probs", {})

        results.append({
            "signal_id": signal.signal_id,
            "true_label": true_label,
            "agent_action": action,
            "llm_raw_action": meta.get("llm_raw_action"),
            "route_taken": route,
            "agent_confidence": confidence,
            "llm_confidence": meta.get("self_reported_confidence"),
            "llm_confidence_source": meta.get("llm_confidence_source"),
            "label_match": action == true_label,
            "is_quarantined": action == "QUARANTINE",
            "abstained": bool(meta.get("abstain_reason")),
            "parse_failed": bool(meta.get("parse_failed", False)),
            "guard_applied": bool(meta.get("policy_guard_applied", False)),
            "guard_reasons": " | ".join(meta.get("policy_guard_reasons", [])),
            "review_reasons": " | ".join(meta.get("review_reasons", [])),
            "review_policies_json": json.dumps(meta.get("review_policies", [])),
            "ml_predicted_class": ctx.get("ml_predicted_class"),
            "ml_top_prob": tel["ml_uncertainty"]["ml_top_prob"],
            "ml_prob_valid": probs.get("VALID"), "ml_prob_spam": probs.get("SPAM"),
            "ml_prob_phishing": probs.get("PHISHING"),
            "ml_entropy_bits": tel["ml_uncertainty"]["ml_entropy_bits"],
            "ml_top2_margin": tel["ml_uncertainty"]["ml_top2_margin"],
            "conformal_set": "|".join(ctx.get("conformal_set", [])),
            "conformal_singleton": ctx.get("conformal_singleton"),
            "ioc_hits": ctx.get("ioc_hits", 0),
            "ioc_host_hits": ctx.get("ioc_host_hits", 0),
            "nn_top_sim": ctx.get("nn_top_sim", 0.0),
            "nn_top_label": ctx.get("nn_top_label", ""),
            "neighbors_json": json.dumps([{"label": n["label"], "sim": n["sim"]} for n in ctx.get("neighbors", [])]),
            "rag_similarity": tel["rag"]["rag_top1_similarity"],
            "unsupported_citations": ",".join(tel["grounding"]["unsupported_citations"]),
            "confidence_gap_vs_ml": tel["confidence"]["confidence_gap_vs_ml"],
            "flag_unsupported_citation": tel["flags"]["unsupported_citation"],
            "flag_weak_retrieval": tel["flags"]["weak_retrieval"],
            "flag_high_ml_entropy": tel["flags"]["high_ml_entropy"],
            "retrieval_ms": round(retrieval_ms, 2),
            "graph_ms": round(graph_ms, 2),
            "end_to_end_ms": round(retrieval_ms + graph_ms, 2),   # includes embedding + Milvus, unlike the old figure
        })

    df = pd.DataFrame(results)
    out_csv = run_dir / f"{stem}.csv"
    df.to_csv(out_csv, index=False)

    summary = summarize(df)
    md = to_markdown(summary, title=f"Evaluation {run_id} ({mode}, alpha={loader.alpha}, fold={args.fold})")
    md += ("\n## Run metadata\n\n"
           "See [metadata.json](metadata.json) for model parameters, split and data fingerprints, "
           "retrieval and guard settings, and package versions. The LLM model digest is not captured.\n"
           "\n## Latency (ms, per email)\n\n"
           f"- retrieval (embedding + Milvus): mean {df['retrieval_ms'].mean():.1f}\n"
           f"- graph (LLM or bypass node only): mean {df['graph_ms'].mean():.1f}\n"
           f"- end-to-end: mean {df['end_to_end_ms'].mean():.1f}, p95 {df['end_to_end_ms'].quantile(0.95):.1f}\n"
           f"\n## Data integrity\n\n```json\n{json.dumps(leak, indent=1)}\n```\n")
    out_md = out_csv.with_suffix(".md")
    out_md.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved telemetry to `{out_csv}`, report to `{out_md}`, and run metadata to `{metadata_path}`")


if __name__ == "__main__":
    run_evaluation()
