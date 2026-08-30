from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import os
import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from src.domain_interface import ThreatSignal, DecisionOutput, DomainAdapter

logger = logging.getLogger(__name__)

class SecurityTools:
    """Domain-agnostic action execution."""
    
    def __init__(self, audit_log_path: str = 'logs/audit.log'):
        self.audit_log_path = audit_log_path
        Path(audit_log_path).parent.mkdir(exist_ok=True)
        
        self.stats = {
            'blocked': 0,
            'flagged': 0,
            'allowed': 0,
            'retrains_triggered': 0,
            'errors': 0
        }
    
    def block(self, signal_id: str, reason: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "BLOCK", reason, audit_id)
        self.stats['blocked'] += 1
        return audit_id
    
    def flag(self, signal_id: str, reason: str, priority: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "FLAG", reason, audit_id)
        self.stats['flagged'] += 1
        return audit_id
    
    def allow(self, signal_id: str, confidence: float) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "ALLOW", f"confidence={confidence:.2%}", audit_id)
        self.stats['allowed'] += 1
        return audit_id
    
    def trigger_retrain(self, signal_id: str, reason: str, urgency: str) -> str:
        audit_id = str(uuid.uuid4())
        self._log_audit(signal_id, "RETRAIN", reason, audit_id)
        self.stats['retrains_triggered'] += 1
        return audit_id
    
    def _log_audit(self, signal_id: str, action: str, details: str, audit_id: str):
        entry = {
            'audit_id': audit_id,
            'timestamp': datetime.utcnow().isoformat(),
            'signal_id': signal_id,
            'action': action,
            'details': details
        }
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        return audit_id
    
    def get_stats(self):
        return self.stats.copy()


class ThreatResponderAgent:
    """
    Universal LLM-based decision agent.
    Works with ANY domain adapter that implements DomainAdapter interface.
    """
    
    def __init__(self, tools: SecurityTools = None, 
                 llm_api_key: str = None, temperature: float = 0.1):
        self.tools = tools or SecurityTools()
        
        if llm_api_key:
            os.environ['OPENAI_API_KEY'] = llm_api_key
        
        self.llm = OpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo-instruct",
            max_tokens=250
        )
        
        self._setup_tools()
        self.agent = initialize_agent(
            self.tool_list,
            self.llm,
            agent="zero-shot-react-description",
            verbose=False
        )
    
    def _setup_tools(self):
        self.tool_list = [
            Tool(
                name="Block Signal",
                func=lambda args: self.tools.block(**args),
                description="Block high-risk signal. Args: signal_id, reason."
            ),
            Tool(
                name="Flag for Review",
                func=lambda args: self.tools.flag(**args),
                description="Flag suspicious signal for human review. Args: signal_id, reason, priority."
            ),
            Tool(
                name="Allow",
                func=lambda args: self.tools.allow(**args),
                description="Allow low-risk signal. Args: signal_id, confidence."
            ),
            Tool(
                name="Trigger Retrain",
                func=lambda args: self.tools.trigger_retrain(**args),
                description="Schedule model retraining. Args: signal_id, reason, urgency."
            )
        ]
    
    def respond(self, signal: ThreatSignal, adapter: DomainAdapter) -> DecisionOutput:
        """
        Main decision method. Works with any domain adapter.
        
        Args:
            signal: Domain-agnostic threat signal
            adapter: Domain-specific adapter for context building
        
        Returns:
            DecisionOutput with standardized action and metadata
        """
        # Build domain-specific prompt
        prompt_template = adapter.get_prompt_template() or self._default_prompt
        prompt = self._build_prompt(signal, adapter, prompt_template)
        
        try:
            from time import perf_counter
            start = perf_counter()
            
            # Get agent decision
            response = self.agent.run(prompt)
            
            # Parse and execute
            action = self._parse_action(response)
            audit_id = self._execute_action(signal.signal_id, signal.risk_score, 
                                          signal.confidence, action)
            
            latency = (perf_counter() - start) * 1000
            
            return DecisionOutput(
                action=action,
                reasoning=response[:200],
                audit_id=audit_id,
                latency_ms=round(latency, 2),
                metadata={'domain': adapter.domain_name}
            )
        except Exception as e:
            logger.warning(f"Agent failed, using fallback: {e}")
            return self._fallback_decision(signal, adapter, str(e))
    
    def _default_prompt(self, signal: ThreatSignal, adapter: DomainAdapter) -> str:
        ctx_lines = "\n".join(f"  - {k}: {v}" for k, v in signal.context.items()[:8])
        return f"""
You are an autonomous security response agent for {adapter.domain_description}.

THREAT ANALYSIS:
- Risk Score: {signal.risk_score:.2%}
- Model Confidence: {signal.confidence:.2f}
- Domain: {adapter.domain_name}
- Signal ID: {signal.signal_id}
- Context:
{ctx_lines}

Available Actions:
1. BLOCK    - High risk ({adapter.domain_name}), act immediately
2. FLAG     - Moderate risk, escalate for human review
3. ALLOW    - Low risk, permit normal operation
4. RETRAIN  - Systemic issue detected, schedule model update

Choose action with reasoning.
Format:
ACTION: [BLOCK|FLAG|ALLOW|RETRAIN]
REASONING: [one sentence]
"""
    
    def _build_prompt(self, signal: ThreatSignal, adapter: DomainAdapter, 
                     prompt_template: callable) -> str:
        if callable(prompt_template):
            return prompt_template(signal, adapter)
        return prompt_template
    
    def _parse_action(self, response: str) -> str:
        response_lower = response.lower()
        
        if 'block' in response_lower:
            return 'BLOCK'
        elif 'flag' in response_lower:
            return 'FLAG'
        elif 'allow' in response_lower:
            return 'ALLOW'
        elif 'retrain' in response_lower:
            return 'RETRAIN'
        else:
            return 'FLAG'  # Safe default
    
    def _execute_action(self, signal_id: str, risk: float, 
                       confidence: float, action: str) -> str:
        if action == 'BLOCK':
            return self.tools.block(signal_id, f"HIGH RISK ({risk:.2%})")
        elif action == 'FLAG':
            priority = 'high' if risk > 0.85 else 'medium'
            return self.tools.flag(signal_id, f"Moderate risk ({risk:.2%})", priority)
        elif action == 'ALLOW':
            return self.tools.allow(signal_id, 1 - risk)
        elif action == 'RETRAIN':
            return self.tools.trigger_retrain(signal_id, "Performance degradation", "normal")
        else:
            return self.tools.flag(signal_id, "Unknown action, defaulting", "low")
    
    def _fallback_decision(self, signal: ThreatSignal, adapter: DomainAdapter, 
                          error_msg: str) -> DecisionOutput:
        """Deterministic rules if LLM agent fails."""
        if signal.risk_score > 0.90:
            action = 'BLOCK'
        elif signal.risk_score > 0.70 or signal.confidence < 0.60:
            action = 'FLAG'
        else:
            action = 'ALLOW'
        
        audit_id = self._execute_action(signal.signal_id, signal.risk_score, 
                                       signal.confidence, action)
        
        return DecisionOutput(
            action=action,
            reasoning=f"Fallback due to agent error: {error_msg[:50]}",
            audit_id=audit_id,
            latency_ms=0,
            metadata={'domain': adapter.domain_name, 'fallback': True}
        )
    
    def get_stats(self):
        return self.tools.get_stats()