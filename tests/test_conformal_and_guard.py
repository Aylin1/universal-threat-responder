import numpy as np

from src.conformal import SplitConformalRouter
from src.guard import GuardConfig, evaluate_guard

CLASSES = ["VALID", "SPAM", "PHISHING"]


def _synthetic(n, seed, sharpness=2.0):
    """Probabilities that are informative but imperfect (labels drawn from the stated probabilities'
    own generator, so the exchangeability assumption holds between calibration and test)."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.5, 0.3, 0.2])
    logits = rng.normal(size=(n, 3))
    logits[np.arange(n), y] += sharpness
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return p, y


def test_marginal_coverage_meets_target():
    pc, yc = _synthetic(3000, 0)
    pt, yt = _synthetic(6000, 1)
    r = SplitConformalRouter(CLASSES, alpha=0.1, class_conditional=False).fit(pc, yc)
    rep = r.evaluate(pt, yt)
    assert rep["coverage"] >= 0.9 - 0.02  # finite-sample slack


def test_class_conditional_coverage_per_class():
    pc, yc = _synthetic(6000, 2)
    pt, yt = _synthetic(12000, 3)
    r = SplitConformalRouter(CLASSES, alpha=0.1, class_conditional=True).fit(pc, yc)
    rep = r.evaluate(pt, yt)
    for c, cov in rep["coverage_per_class"].items():
        assert cov >= 0.9 - 0.03, (c, cov)


def test_smaller_alpha_gives_larger_sets():
    pc, yc = _synthetic(3000, 4)
    pt, yt = _synthetic(3000, 5)
    tight = SplitConformalRouter(CLASSES, alpha=0.2).fit(pc, yc).evaluate(pt, yt)
    loose = SplitConformalRouter(CLASSES, alpha=0.01).fit(pc, yc).evaluate(pt, yt)
    assert loose["mean_set_size"] >= tight["mean_set_size"]
    assert loose["singleton_rate"] <= tight["singleton_rate"]


def test_tiny_class_includes_class_instead_of_overclaiming():
    # only 5 calibration points for SPAM, alpha=0.05 needs >= 19 -> qhat=inf -> SPAM always in the set
    p = np.array([[0.9, 0.05, 0.05]] * 50 + [[0.1, 0.8, 0.1]] * 5)
    y = np.array([0] * 50 + [1] * 5)
    r = SplitConformalRouter(CLASSES, alpha=0.05).fit(p, y)
    assert np.isinf(r.qhat["SPAM"])
    d = r.predict_one([0.95, 0.03, 0.02])
    assert "SPAM" in d.prediction_set and not d.is_singleton  # forced to escalate


def test_singleton_flag_and_unfitted_guard():
    r = SplitConformalRouter(CLASSES, alpha=0.5)
    try:
        r.predict_one([0.3, 0.3, 0.4])
        assert False, "should require fit"
    except RuntimeError:
        pass


# ----------------------------------------------------------------------------- guard
def test_guard_only_acts_on_benign_verdicts():
    v = evaluate_guard("PHISHING", ml_class="VALID", ml_prob=0.99, neighbors=[], ioc_hits=3)
    assert v.action == "PHISHING" and not v.applied


def test_guard_never_upgrades_to_allow():
    v = evaluate_guard("QUARANTINE", ml_class="VALID", ml_prob=0.99, neighbors=[])
    assert v.action == "QUARANTINE"


def test_guard_ioc_always_vetoes():
    v = evaluate_guard("VALID", ml_class="VALID", ml_prob=0.99, neighbors=[], ioc_hits=1)
    assert v.action == "QUARANTINE" and v.applied


def test_guard_ml_veto_threshold():
    kw = dict(neighbors=[], ioc_hits=0)
    assert evaluate_guard("VALID", ml_class="SPAM", ml_prob=0.7, **kw).applied
    assert not evaluate_guard("VALID", ml_class="SPAM", ml_prob=0.4, **kw).applied
    assert not evaluate_guard("VALID", ml_class="VALID", ml_prob=0.99, **kw).applied


def test_guard_neighbor_veto_needs_relevant_and_enough_neighbors():
    cfg = GuardConfig(enable_ml_veto=False)
    low_sim = [{"label": "PHISHING", "sim": 0.58}] * 8   # the old failure mode: many weak matches
    assert not evaluate_guard("VALID", ml_class="VALID", ml_prob=0.9, neighbors=low_sim, cfg=cfg).applied
    one = [{"label": "PHISHING", "sim": 0.95}]
    assert not evaluate_guard("VALID", ml_class="VALID", ml_prob=0.9, neighbors=one, cfg=cfg).applied
    strong = [{"label": "PHISHING", "sim": 0.9}, {"label": "SPAM", "sim": 0.85}, {"label": "VALID", "sim": 0.8}]
    assert evaluate_guard("VALID", ml_class="VALID", ml_prob=0.9, neighbors=strong, cfg=cfg).applied


def test_guard_neighbor_majority_benign_does_not_veto():
    cfg = GuardConfig(enable_ml_veto=False)
    nb = [{"label": "VALID", "sim": 0.9}] * 4 + [{"label": "SPAM", "sim": 0.9}]
    assert not evaluate_guard("VALID", ml_class="VALID", ml_prob=0.9, neighbors=nb, cfg=cfg).applied
