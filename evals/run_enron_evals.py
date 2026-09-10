import pandas as pd
from pathlib import Path
from langchain_ollama import ChatOllama
from src.core import ThreatResponderGraph, SecurityTools
from adapters.rag_spam_adapter import RAGSpamAdapter
from adapters.enron_loader import EnronDatasetAdapter
from evals.llm_judge import LLMJudgeEvaluator


def run_evaluation():
    print("Initializing LLM and Agent Graph...")
    llm = ChatOllama(model="llama3", temperature=0.1, keep_alive=-1)

    tools = SecurityTools(audit_log_path="logs/eval_audit.log")
    agent = ThreatResponderGraph(llm=llm, tools=tools)
    evaluator = LLMJudgeEvaluator(judge_llm=llm)

    adapter = RAGSpamAdapter(vector_db_path="data/chroma_db")
    
    # Use EnronDatasetAdapter to load and structure test items
    dataset_loader = EnronDatasetAdapter(file_path="data/spam/enron_spam_data.csv")
    test_items = dataset_loader.load_as_test_items(sample_fraction=0.01, seed=42)

    results = []
    print(f"\nRunning evaluation across {len(test_items)} samples with RAGSpamAdapter...")

    for item in test_items:
        email_text = item["input"]
        true_label = item["expected_action"]

        # 1. Build RAG Context
        context = adapter.build_context(raw_input=email_text, metadata={"source": "enron_eval"})

        # 2. Build ThreatSignal via adapter's signal builder
        signal = item["signal_builder"](context)

        # 3. Execute Graph
        try:
            decision = agent.respond(signal=signal, adapter=adapter)
            agent_action = decision.action.upper()
            agent_reasoning = decision.reasoning
        except Exception as e:
            agent_action = "FLAG"
            agent_reasoning = f"Graph execution failed: {str(e)}"

        # 4. Audit with LLM Judge
        audit = evaluator.evaluate_response(
            email_text=email_text,
            policies=context.get('retrieved_policies', []),
            agent_reasoning=agent_reasoning,
            action_taken=agent_action
        )

        results.append({
            "signal_id": signal.signal_id,
            "true_label": true_label,
            "agent_action": agent_action,
            "label_match": (agent_action == true_label),
            "faithfulness": audit.faithfulness_score,
            "correctness": audit.correctness_score,
            "is_hallucinated": audit.is_hallucinated,
            "audit_explanation": audit.explanation
        })

    # Output report
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 45)
    print("📊 LIVE LANGGRAPH + RAG SECURITY & AUDIT REPORT")
    print("=" * 45)
    print(f"Total Evaluated: {len(results_df)}")
    print(f"Dataset Accuracy Match: {results_df['label_match'].mean() * 100:.2f}%")
    print(f"Average Judge Faithfulness: {results_df['faithfulness'].mean():.2f} / 1.0")
    print(f"Average Judge Correctness: {results_df['correctness'].mean():.2f} / 1.0")
    print(f"Hallucination Rate: {results_df['is_hallucinated'].mean() * 100:.2f}%")
    print("=" * 45)

    output_path = Path("evals/live_audit_report_results.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved detailed audit trail to `{output_path}`")


if __name__ == "__main__":
    run_evaluation()