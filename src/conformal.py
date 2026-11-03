"""Split conformal prediction for routing.

Replaces the previous `max_prob >= 0.95` rule, which was a confidence threshold and
carried no coverage guarantee.

Method (LAC / "least ambiguous" score, optionally class-conditional / Mondrian):
  * nonconformity score of (x, y):  s = 1 - p_y(x)
  * on a held-out CALIBRATION fold, take the finite-sample-corrected quantile
        qhat = the ceil((n+1)(1-alpha))-th smallest score
    (per true class when class_conditional=True)
  * prediction set for a new x:  { k : 1 - p_k(x) <= qhat_k }
  * route to the fast path only when the set is a SINGLETON; empty or multi-label sets
    are ambiguous and escalate to the LLM.

Guarantee (exchangeability): P(y in set) >= 1 - alpha marginally, or per class when
class_conditional=True. Note this bounds coverage of the *set*, it does not by itself
guarantee accuracy among singleton predictions, so accuracy-on-singletons is reported
empirically in `evaluate`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ConformalDecision:
    prediction_set: List[str]
    is_singleton: bool
    predicted_class: Optional[str]  # populated only for singletons


class SplitConformalRouter:
    def __init__(self, classes: Sequence[str], alpha: float = 0.05, class_conditional: bool = True):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.classes = list(classes)
        self.alpha = float(alpha)
        self.class_conditional = bool(class_conditional)
        self.qhat: Dict[str, float] = {}
        self.class_counts: Dict[str, int] = {}
        self.n_cal = 0
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def _quantile(self, scores: np.ndarray) -> float:
        n = len(scores)
        if n == 0:
            return math.inf
        level = math.ceil((n + 1) * (1.0 - self.alpha))
        if level > n:  # not enough calibration points for this alpha -> include the class always
            return math.inf
        return float(np.sort(scores)[level - 1])

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "SplitConformalRouter":
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        if probs.ndim != 2 or probs.shape[1] != len(self.classes):
            raise ValueError("probs must be (n, n_classes)")
        scores = 1.0 - probs[np.arange(len(y)), y]
        self.n_cal = int(len(y))
        global_q = self._quantile(scores)
        for k, c in enumerate(self.classes):
            s_k = scores[y == k]
            self.class_counts[c] = int(len(s_k))
            self.qhat[c] = self._quantile(s_k) if self.class_conditional else global_q
        self._fitted = True
        return self

    # -------------------------------------------------------------- predict
    def predict_one(self, probs_row: Sequence[float]) -> ConformalDecision:
        if not self._fitted:
            raise RuntimeError("Router not fitted; call fit() on a calibration fold first")
        p = np.asarray(probs_row, dtype=float)
        pset = [c for k, c in enumerate(self.classes) if (1.0 - p[k]) <= self.qhat[c]]
        single = len(pset) == 1
        return ConformalDecision(pset, single, pset[0] if single else None)

    def predict(self, probs: np.ndarray) -> List[ConformalDecision]:
        return [self.predict_one(r) for r in np.asarray(probs, dtype=float)]

    # ------------------------------------------------------------- evaluate
    def evaluate(self, probs: np.ndarray, y: np.ndarray) -> Dict[str, object]:
        """Empirical coverage / efficiency on data NOT used for calibration."""
        y = np.asarray(y, dtype=int)
        decisions = self.predict(probs)
        covered = np.array([self.classes[t] in d.prediction_set for t, d in zip(y, decisions)])
        sizes = np.array([len(d.prediction_set) for d in decisions])
        single = sizes == 1
        single_correct = np.array([d.is_singleton and d.predicted_class == self.classes[t]
                                   for t, d in zip(y, decisions)])
        per_class = {c: float(covered[y == k].mean()) if (y == k).any() else float("nan")
                     for k, c in enumerate(self.classes)}
        return {
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "coverage": float(covered.mean()),
            "coverage_per_class": per_class,
            "singleton_rate": float(single.mean()),
            "empty_rate": float((sizes == 0).mean()),
            "multi_rate": float((sizes > 1).mean()),
            "singleton_accuracy": float(single_correct[single].mean()) if single.any() else float("nan"),
            "mean_set_size": float(sizes.mean()),
            "n_eval": int(len(y)),
            "n_cal": self.n_cal,
        }

    def to_dict(self) -> Dict[str, object]:
        return {"classes": self.classes, "alpha": self.alpha, "class_conditional": self.class_conditional,
                "qhat": {k: (None if math.isinf(v) else v) for k, v in self.qhat.items()},
                "class_counts": self.class_counts, "n_cal": self.n_cal}
