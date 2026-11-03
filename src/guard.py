"""Policy guard (pure function, no I/O).

Old behaviour: veto an LLM `VALID` whenever ANY retrieved document carried a static
metadata risk_score >= 0.95. Because the 30 OpenPhish documents all carry 0.95 and share
boilerplate text that dominates their embeddings, this fired on ~75% of samples regardless
of how relevant the match was.

New behaviour: the veto is driven by evidence that is actually about THIS email:
  1. exact IOC hit (URL in a threat feed)                       -> always veto
  2. the local ML model confidently predicts a non-benign class -> veto
  3. >= N labelled historical emails that are genuinely similar
     (cosine >= neighbor_min_sim) and mostly non-benign         -> veto

The guard is deliberately ASYMMETRIC: it can only turn a benign verdict into QUARANTINE.
It never turns a block/spam verdict into an allow, so its failure mode is abstention.

The default thresholds are starting points, not tuned values. Use evals/sweep_guard.py to
tune them on a dev run and confirm on a different fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class GuardConfig:
    benign_action: str = "VALID"
    quarantine_action: str = "QUARANTINE"
    enable_ml_veto: bool = True
    ml_veto_min_prob: float = 0.60
    enable_neighbor_veto: bool = True
    neighbor_min_sim: float = 0.75
    neighbor_min_count: int = 2
    neighbor_min_nonbenign_frac: float = 0.60


@dataclass
class GuardVerdict:
    action: str
    applied: bool = False
    reasons: List[str] = field(default_factory=list)


def evaluate_guard(
    llm_action: str,
    *,
    ml_class: str,
    ml_prob: float,
    neighbors: Sequence[Dict[str, Any]],
    ioc_hits: int = 0,
    cfg: GuardConfig = GuardConfig(),
) -> GuardVerdict:
    """neighbors: [{"label": "SPAM", "sim": 0.81}, ...] (labels already normalised)."""
    if str(llm_action).upper() != cfg.benign_action:
        return GuardVerdict(action=llm_action)

    reasons: List[str] = []

    if ioc_hits > 0:
        reasons.append(f"{ioc_hits} indicator(s) matched a live threat feed")

    if cfg.enable_ml_veto and ml_class and ml_class != cfg.benign_action and float(ml_prob) >= cfg.ml_veto_min_prob:
        reasons.append(f"local ML predicts {ml_class} (p={float(ml_prob):.2f} >= {cfg.ml_veto_min_prob:.2f})")

    if cfg.enable_neighbor_veto:
        relevant = [n for n in neighbors if float(n.get("sim", 0.0)) >= cfg.neighbor_min_sim]
        if len(relevant) >= cfg.neighbor_min_count:
            bad = sum(1 for n in relevant if n.get("label") != cfg.benign_action)
            frac = bad / len(relevant)
            if frac >= cfg.neighbor_min_nonbenign_frac:
                reasons.append(
                    f"{bad}/{len(relevant)} similar labelled emails (cos>={cfg.neighbor_min_sim:.2f}) are non-benign"
                )

    if reasons:
        return GuardVerdict(action=cfg.quarantine_action, applied=True, reasons=reasons)
    return GuardVerdict(action=llm_action)
