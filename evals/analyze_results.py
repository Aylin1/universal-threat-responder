"""Offline analysis of an evaluation CSV (no LLM, no Milvus needed).

    python -m evals.analyze_results "evals/results/historical/live_audit_report_results_*.csv"

Why this exists: the old headline ("accuracy 85.8% -> 90.0%") was accuracy on RESOLVED rows only,
so enabling the fast path partly changed WHICH rows were scored. This report always shows, side
by side, for every system:

  decided/coverage    share of emails that received a non-QUARANTINE decision
  resolved accuracy   accuracy on decided emails only          (the old headline)
  end-to-end accuracy accuracy over ALL emails, abstention = wrong   (comparable across systems)
  leak rate           spam/phishing that was auto-ALLOWED       (the costly security error)
  false-block rate    valid mail that was auto-BLOCKED/SPAM'd

and it includes the baseline the old report was missing: the local ML model alone.
"""
import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.metrics import bootstrap_ci, brier_score, expected_calibration_error

LABELS = ["VALID", "SPAM", "PHISHING"]
ABSTAIN = "QUARANTINE"


# --------------------------------------------------------------------------- helpers
def _as_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in {"true", "1", "1.0", "yes"})


def _system_metrics(true: pd.Series, pred: pd.Series, name: str, seed: int = 0) -> Dict[str, Any]:
    n = len(true)
    decided = pred != ABSTAIN
    correct = (pred == true) & decided
    e2e, e2e_lo, e2e_hi = bootstrap_ci(correct.astype(float).values, seed=seed)
    resolved = float(correct[decided].mean()) if decided.any() else float("nan")
    r_mean, r_lo, r_hi = bootstrap_ci(correct[decided].astype(float).values, seed=seed) if decided.any() \
        else (float("nan"),) * 3
    nonbenign = true != "VALID"
    leak = ((pred == "VALID") & nonbenign).sum() / max(1, nonbenign.sum())
    benign = true == "VALID"
    false_block = (decided & (pred != "VALID") & benign).sum() / max(1, benign.sum())
    return {
        "system": name, "n": n,
        "coverage": float(decided.mean()),
        "resolved_accuracy": resolved, "resolved_ci": (r_lo, r_hi),
        "end_to_end_accuracy": e2e, "end_to_end_ci": (e2e_lo, e2e_hi),
        "leak_rate": float(leak), "false_block_rate": float(false_block),
    }


def per_class_report(true: pd.Series, pred: pd.Series) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in LABELS:
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        out[c] = {"precision": tp / (tp + fp) if tp + fp else float("nan"),
                  "recall_on_decided": tp / (tp + fn) if tp + fn else float("nan"),
                  "support": int((true == c).sum())}
    return out


def confusion_with_abstain(true: pd.Series, pred: pd.Series) -> pd.DataFrame:
    cols = LABELS + [ABSTAIN]
    return pd.crosstab(true, pred).reindex(index=LABELS, columns=cols, fill_value=0)


# --------------------------------------------------------------------------- main summary
def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    for col in ("guard_applied", "abstained", "conformal_singleton"):
        if col in df.columns:
            df[col] = _as_bool(df[col])
    true, pred = df["true_label"], df["agent_action"]
    route = df["route_taken"].astype(str)
    out: Dict[str, Any] = {"n": int(len(df))}

    systems: List[Dict[str, Any]] = []
    if "ml_predicted_class" in df.columns and df["ml_predicted_class"].isin(LABELS).all():
        ml = df["ml_predicted_class"]
        systems.append(_system_metrics(true, ml, "ML only (argmax, no abstention)"))
        if "conformal_singleton" in df.columns:
            singleton_prediction = df["conformal_set"] if "conformal_set" in df.columns else ml
            ml_abs = singleton_prediction.where(df["conformal_singleton"] & singleton_prediction.isin(LABELS), ABSTAIN)
            systems.append(_system_metrics(true, ml_abs, "ML + conformal (abstain unless singleton set)"))
    else:
        out["ml_baseline_unavailable"] = True
    full_name = ("FULL: agreement review + RAG/LLM" if "review_reasons" in df.columns
                 else "FULL: conformal fast-path + RAG/LLM + guard")
    systems.append(_system_metrics(true, pred, full_name))

    if "guard_applied" in df.columns and "llm_raw_action" in df.columns and "review_reasons" not in df.columns:
        no_guard = pred.copy()
        mask = df["guard_applied"] & df["llm_raw_action"].notna()
        no_guard[mask] = df.loc[mask, "llm_raw_action"]
        systems.append(_system_metrics(true, no_guard, "FULL without the policy guard (counterfactual)"))
    out["systems"] = systems

    # ---- per-route breakdown
    routes = []
    for r in sorted(route.unique()):
        m = route == r
        routes.append({**_system_metrics(true[m], pred[m], r), "share": float(m.mean())})
    out["routes"] = routes

    # ---- per-class + confusion on the full system
    dec = pred != ABSTAIN
    out["per_class_resolved"] = per_class_report(true[dec], pred[dec]) if dec.any() else {}
    out["confusion"] = confusion_with_abstain(true, pred)

    # ---- coverage-accuracy curve for the ML model alone (threshold sweep)
    if not out.get("ml_baseline_unavailable") and "ml_top_prob" in df.columns and df["ml_top_prob"].notna().all():
        curve = []
        for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]:
            m = df["ml_top_prob"] >= t
            if m.any():
                curve.append({"threshold": t, "coverage": float(m.mean()),
                              "accuracy": float((df.loc[m, "ml_predicted_class"] == true[m]).mean())})
        out["ml_coverage_curve"] = curve

    # ---- calibration against CORRECTNESS (not against another model)
    cal: Dict[str, Any] = {}
    if not out.get("ml_baseline_unavailable") and "ml_top_prob" in df.columns and df["ml_top_prob"].notna().all():
        ok = (df["ml_predicted_class"] == true).values
        ece, _ = expected_calibration_error(df["ml_top_prob"].values, ok)
        cal["ml_ece"], cal["ml_brier"] = ece, brier_score(df["ml_top_prob"].values, ok)
    if "llm_raw_action" in df.columns:
        m = df["llm_raw_action"].notna() & (df["llm_raw_action"].astype(str) != "")
        if m.any():
            ok = (df.loc[m, "llm_raw_action"] == true[m]).values
            conf = df.loc[m, "agent_confidence"].astype(float).values
            ece, _ = expected_calibration_error(conf, ok)
            cal["llm_verbalized_ece"] = ece
            cal["llm_verbalized_brier"] = brier_score(conf, ok)
            cal["llm_raw_accuracy"] = float(ok.mean())
            cal["llm_n"] = int(m.sum())
    out["calibration"] = cal

    # ---- policy-guard impact (only meaningful on rows where the LLM said VALID)
    if "guard_applied" in df.columns and "llm_raw_action" in df.columns and "review_reasons" not in df.columns:
        v = df["llm_raw_action"] == "VALID"
        if v.any():
            vetoed = v & df["guard_applied"]
            out["guard"] = {
                "llm_said_valid": int(v.sum()),
                "vetoed": int(vetoed.sum()),
                "veto_rate": float(vetoed.sum() / v.sum()),
                "vetoed_but_actually_valid": int((vetoed & (true == "VALID")).sum()),   # cost of the guard
                "vetoed_and_actually_bad": int((vetoed & (true != "VALID")).sum()),     # benefit of the guard
                "unvetoed_but_actually_bad": int((v & ~df["guard_applied"] & (true != "VALID")).sum()),  # leaks
            }
    return out


# --------------------------------------------------------------------------- rendering
def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def _ci(ci) -> str:
    return "" if ci[0] != ci[0] else f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _df_md(df: pd.DataFrame) -> List[str]:
    """Render a small DataFrame as a markdown table without needing `tabulate`."""
    cols = list(df.columns)
    lines = ["| truth \\ pred | " + " | ".join(map(str, cols)) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(int(row[c])) for c in cols) + " |")
    return lines


def to_markdown(s: Dict[str, Any], title: str = "Benchmark results (generated)") -> str:
    L = [f"# {title}", "", f"Samples: {s['n']}", "",
         "## System comparison", "",
         "`End-to-end` counts abstentions (QUARANTINE) as wrong, so it is comparable across systems. "
         "`Resolved` is accuracy only on emails the system decided. 95% bootstrap CIs in brackets.", "",
         "| System | Coverage | Resolved acc. | End-to-end acc. | Leak rate | False-block rate |",
         "|---|---|---|---|---|---|"]
    for m in s["systems"]:
        L.append(f"| {m['system']} | {_pct(m['coverage'])} | {_pct(m['resolved_accuracy'])}{_ci(m['resolved_ci'])} | "
                 f"{_pct(m['end_to_end_accuracy'])}{_ci(m['end_to_end_ci'])} | {_pct(m['leak_rate'])} | "
                 f"{_pct(m['false_block_rate'])} |")
    if s.get("ml_baseline_unavailable"):
        L += ["", "ML baseline omitted: one or more rows have missing or invalid ML predictions."]
    L += ["", "Leak rate = spam/phishing auto-ALLOWED / all spam+phishing. "
          "False-block rate = valid mail auto-blocked or spam-filtered / all valid mail.", "",
          "## By route", "", "| Route | Share | Coverage | Resolved acc. | End-to-end acc. |", "|---|---|---|---|---|"]
    for m in s["routes"]:
        L.append(f"| {m['system']} | {_pct(m['share'])} | {_pct(m['coverage'])} | "
                 f"{_pct(m['resolved_accuracy'])} | {_pct(m['end_to_end_accuracy'])} |")

    L += ["", "## Confusion matrix (full system; rows = truth)", ""] + _df_md(s["confusion"]) + [""]
    if s.get("per_class_resolved"):
        L += ["## Per-class (decided emails only)", "", "| Class | Precision | Recall (decided) | Support |", "|---|---|---|---|"]
        for c, m in s["per_class_resolved"].items():
            L.append(f"| {c} | {_pct(m['precision'])} | {_pct(m['recall_on_decided'])} | {m['support']} |")
        L.append("")
    if s.get("ml_coverage_curve"):
        L += ["## ML-only coverage vs accuracy", "", "| Min top prob | Coverage | Accuracy |", "|---|---|---|"]
        for r in s["ml_coverage_curve"]:
            L.append(f"| {r['threshold']:.2f} | {_pct(r['coverage'])} | {_pct(r['accuracy'])} |")
        L.append("")
    if s.get("calibration"):
        L += ["## Calibration (against correctness)", ""]
        for k, v in s["calibration"].items():
            L.append(f"- `{k}`: {v:.4f}" if isinstance(v, float) else f"- `{k}`: {v}")
        L.append("")
    if s.get("guard"):
        L += ["## Policy-guard impact (rows where the LLM said VALID)", ""]
        for k, v in s["guard"].items():
            L.append(f"- `{k}`: {v:.3f}" if isinstance(v, float) else f"- `{k}`: {v}")
        L.append("")
    return "\n".join(L)


def load_many(patterns: List[str]) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise FileNotFoundError(f"No CSV matched {patterns}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="evals/results/generated/benchmark_results_generated.md")
    a = ap.parse_args()
    summary = summarize(load_many(a.csv))
    md = to_markdown(summary)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWrote {a.out}")
