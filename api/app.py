from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Optional
import joblib
from src.core import ThreatResponderAgent, SecurityTools
from src.domain_interface import ThreatSignal
from adapters.spam import SpamAdapter
from adapters.login import LoginAdapter
from time import perf_counter

app = FastAPI(title="Universal Threat Responder")

# Global state
agent = None
tools = None
adapters = {}

# Request schemas
class SpamRequest(BaseModel):
    text: str
    signal_id: Optional[str] = None
    sender_domain: Optional[str] = "unknown"
    message_age_hours: Optional[int] = 0

class LoginRequest(BaseModel):
    signal_id: Optional[str] = None
    ip_distance_km: float
    device_new: bool
    time_since_last_login_hours: Optional[float] = 0
    failed_attempts_24h: Optional[int] = 0
    geo_velocity_kmh: Optional[float] = 0
    user_agent: Optional[str] = None

# Response schema
class DecisionResponse(BaseModel):
    signal_id: str
    domain: str
    risk_score: float
    confidence: float
    action: str
    reasoning: str
    audit_id: str
    latency_ms: float

@app.on_event("startup")
async def startup():
    global agent, tools, adapters
    tools = SecurityTools()
    agent = ThreatResponderAgent(tools)
    
    # Load spam model if available
    try:
        spam_model = joblib.load('models/spam_classifier.pkl')
        spam_vectorizer = joblib.load('models/vectorizer.pkl')
        adapters['spam'] = SpamAdapter()
    except FileNotFoundError:
        pass
    
    adapters['login'] = LoginAdapter()

@app.post("/decide/spam", response_model=DecisionResponse)
async def decide_spam(request: SpamRequest):
    """Email spam decision endpoint."""
    if 'spam' not in adapters:
        raise HTTPException(status_code=503, detail="Spam model not loaded")
    
    adapter = adapters['spam']
    adapter.validate_input(request.text)
    
    # Extract features (dummy ML call here)
    # In production: load real model and predict
    X_text, X_struct, _ = adapter.extract_features(request.text)
    # y_pred_proba = spam_model.predict_proba(X_combined)[0][1]  # Uncomment with real model
    risk_score = 0.5  # Placeholder - replace with actual prediction
    confidence = 0.8
    
    # Build signal
    context = adapter.build_context(request.text, {
        'sender_domain': request.sender_domain,
        'message_age_hours': request.message_age_hours
    })
    
    signal = ThreatSignal(
        signal_id=request.signal_id or 'anon',
        risk_score=risk_score,
        confidence=confidence,
        context=context,
        raw_payload=request.text
    )
    
    # Agent decision
    start = perf_counter()
    decision = agent.respond(signal, adapter)
    latency = (perf_counter() - start) * 1000
    
    return DecisionResponse(
        signal_id=signal.signal_id,
        domain=adapter.domain_name,
        risk_score=risk_score,
        confidence=confidence,
        action=decision.action,
        reasoning=decision.reasoning,
        audit_id=decision.audit_id,
        latency_ms=latency
    )

@app.post("/decide/login", response_model=DecisionResponse)
async def decide_login(request: LoginRequest):
    """Login anomaly decision endpoint."""
    adapter = adapters['login']
    
    # Build raw dict
    raw = {
        'ip_distance_km': request.ip_distance_km,
        'device_new': request.device_new,
        'time_since_last_login_hours': request.time_since_last_login_hours,
        'failed_attempts_24h': request.failed_attempts_24h,
        'geo_velocity_kmh': request.geo_velocity_kmh,
        'user_agent': request.user_agent
    }
    
    adapter.validate_input(raw)
    
    # Extract features (dummy ML call)
    features = adapter.extract_features(raw)
    # risk_score = login_model.predict_proba(features)[0][1]  # Uncomment with real model
    risk_score = 0.3  # Placeholder
    confidence = 0.75
    
    context = adapter.build_context(raw)
    
    signal = ThreatSignal(
        signal_id=request.signal_id or 'anon',
        risk_score=risk_score,
        confidence=confidence,
        context=context,
        raw_payload=raw
    )
    
    start = perf_counter()
    decision = agent.respond(signal, adapter)
    latency = (perf_counter() - start) * 1000
    
    return DecisionResponse(
        signal_id=signal.signal_id,
        domain=adapter.domain_name,
        risk_score=risk_score,
        confidence=confidence,
        action=decision.action,
        reasoning=decision.reasoning,
        audit_id=decision.audit_id,
        latency_ms=latency
    )

@app.get("/stats")
async def get_stats():
    """Return agent action statistics."""
    return {
        'agent_stats': agent.get_stats(),
        'active_adapters': list(adapters.keys()),
        'timestamp': str(datetime.utcnow())
    }

@app.get("/health")
async def health():
    return {'status': 'healthy', 'adapters': list(adapters.keys())}