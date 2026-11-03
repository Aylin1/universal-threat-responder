"""Config-free numeric helpers shared by the core, the evaluator and the analysis scripts.

Nothing in here imports langchain / config, so it can be unit-tested in isolation.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


def compute_shannon_entropy(p: float) -> float:
    """BINARY entropy (bits) of a single probability. Kept for backward compatibility.

    For a 3-class classifier use `distribution_entropy` instead: binary entropy of the
    top-class probability is NOT the entropy of the predictive distribution.
    """
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def distribution_entropy(probs: Iterable[float]) -> float:
    """Shannon entropy (bits) of a full probability vector."""
    p = np.clip(np.asarray(list(probs), dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def l2_to_cosine_similarity(distance: float) -> float:
    """Milvus returns squared L2 for the default metric; for unit-norm vectors cos = 1 - d/2."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def top2_margin(probs: Iterable[float]) -> float:
    p = np.sort(np.asarray(list(probs), dtype=float))[::-1]
    return float(p[0] - p[1]) if len(p) > 1 else float(p[0])


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> Tuple[float, list]:
    """Standard ECE: weighted mean |accuracy - confidence| over equal-width bins.

    Returns (ece, bins) where bins is a list of dicts for reliability diagrams.
    """
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        acc, avg_conf, w = corr[mask].mean(), conf[mask].mean(), mask.mean()
        ece += w * abs(acc - avg_conf)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                     "accuracy": float(acc), "confidence": float(avg_conf)})
    return float(ece), bins


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    conf = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    corr = np.asarray(correct, dtype=float)
    return float(((conf - corr) ** 2).mean()) if len(conf) else float("nan")


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for a mean. Returns (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def confusion_counts(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in out and p in out[t]:
            out[t][p] += 1
    return out
