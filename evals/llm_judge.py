import json
import re
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel


class JudgeEvaluationSchema(BaseModel):
    faithfulness_score: float = Field(
        ...,
        description="Score 0.0-1.0: Does reasoning align with policy without hallucinating rules?"
    )
    correctness_score: float = Field(
        ...,
        description="Score 0.0-1.0: Is the professional judgment sound for enterprise security?"
    )
    is_hallucinated: bool = Field(
        ...,
        description="True if LLM invented non-existent policies or facts."
    )
    explanation: str = Field(
        ...,
        description="One sentence rationale for scores."
    )


class LLMJudgeEvaluator:
    """Evaluates agent responses for hallucinations, professional correctness, and policy alignment."""

    def __init__(self, judge_llm: BaseChatModel):
        # Use raw chat model to avoid fragile LangChain tool-calling wrappers with local models
        self.judge = judge_llm

    def _clean_json_output(self, text: str) -> str:
        """Strips markdown code blocks if the local model wraps its JSON."""
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        return text.strip()

    def evaluate_response(
        self, 
        email_text: str, 
        policies: list,
        agent_reasoning: str,
        action_taken: str
    ) -> JudgeEvaluationSchema:
        policy_str = "\n".join(f"- {p}" for p in policies)

        prompt = f"""SYSTEM: You are a senior AI Security Auditor evaluating an LLM Response Agent. You must output ONLY a raw JSON object matching the exact schema below. Do not include any conversational introduction or explanation outside the JSON.

REQUIRED JSON SCHEMA:
{{
  "faithfulness_score": 0.0 to 1.0 (float),
  "correctness_score": 0.0 to 1.0 (float),
  "is_hallucinated": true or false (boolean),
  "explanation": "One sentence rationale for scores."
}}

CONTEXT POLICIES:
{policy_str}

EVALUATED EMAIL:
\"\"\"
{email_text[:1000]}
\"\"\"

AGENT ACTION TAKEN: {action_taken}
AGENT REASONING: \"\"\"{agent_reasoning}\"\"\"

JSON OUTPUT:
"""
        try:
            response = self.judge.invoke(prompt)
            # Extract content string from LangChain message response
            raw_content = response.content if hasattr(response, "content") else str(response)
            
            cleaned_json = self._clean_json_output(raw_content)
            data = json.loads(cleaned_json)
            
            return JudgeEvaluationSchema(**data)
        except Exception as e:
            return JudgeEvaluationSchema(
                faithfulness_score=0.5,
                correctness_score=0.5,
                is_hallucinated=False,
                explanation=f"Judge evaluation failed to parse: {str(e)}"
            )


if __name__ == "__main__":
    from langchain_ollama import ChatOllama

    print("Initializing local judge model with Ollama (llama3)...")
    llm = ChatOllama(model="llama3", temperature=0.0)
    evaluator = LLMJudgeEvaluator(llm)

    print("Running test evaluation pass...")
    sample_res = evaluator.evaluate_response(
        email_text="Hey team, can we move our project sync meeting to 3 PM today?",
        policies=["Allow internal calendar scheduling and coordination."],
        agent_reasoning="Internal meeting request, safe corporate chatter.",
        action_taken="ALLOW"
    )
    
    print("\n--- Judge Evaluation Output ---")
    print(sample_res.model_dump_json(indent=2))