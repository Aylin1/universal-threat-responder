"""Tune the policy guard offline, without re-running the LLM.

The eval CSV stores the LLM's raw verdict, the ML prediction and the retrieved neighbours, so the
guard (a pure function) can be replayed over a grid of thresholds.

    python -m evals.sweep_guard evals/results/historical/live_audit_report_results_fastpath-disabled_20261102_222648_585601.csv --max-leak 0.02

IMPORTANT: tune on one fold/seed and REPORT on another (`run_evals.py --fold 1`, re-run populate with
the same --fold). Choosing thresholds on the very rows you then report is optimistic.
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.guard import GuardConfig, evaluate_agreement_review, evaluate_guard

ABSTAIN = "QUARANTINE"


def replay_agreement_review(df: pd.DataFrame) -> pd.Series:
    required = {"agent_action", "llm_raw_action", "ml_predicted_class",
                "conformal_set", "conformal_singleton", "ioc_hits"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for agreement review: {sorted(missing)}")
    if df["llm_raw_action"].isna().any() or df["ml_predicted_class"].isna().any():
        raise ValueError("Agreement replay requires ML and LLM predictions for every row; use a fast-path-disabled run")
    if "route_taken" in df.columns and (df["route_taken"] != "ESCALATION_RAG_LLM").any():
        raise ValueError("Agreement replay requires LLM-path results for every row")
    actions = []
    for _, row in df.iterrows():
        raw_set = row["conformal_set"]
        cset = raw_set.split("|") if isinstance(raw_set, str) and raw_set else []
        singleton = str(row["conformal_singleton"]).strip().lower() in {"true", "1", "1.0"}
        verdict = evaluate_agreement_review(
            row["agent_action"], llm_action=row["llm_raw_action"], ml_class=row["ml_predicted_class"],
            conformal_set=cset, conformal_singleton=singleton,
            allowed_actions=("VALID", "SPAM", "PHISHING", "QUARANTINE"),
            quarantine_action=ABSTAIN, ioc_hits=int(row["ioc_hits"]),
        )
        actions.append(verdict.action)
    return pd.Series(actions, index=df.index)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "abstained" in df.columns:
        df["abstained"] = df["abstained"].map(lambda v: str(v).strip().lower() in {"true", "1", "1.0"})
    else:
        df["abstained"] = False
    df = df[df["llm_raw_action"].notna() & (df["llm_raw_action"].astype(str) != "")].reset_index(drop=True)
    df["neighbors"] = df["neighbors_json"].map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
    return df


def simulate(df: pd.DataFrame, cfg: GuardConfig) -> pd.Series:
    out = []
    for _, r in df.iterrows():
        if r["abstained"]:
            out.append(ABSTAIN)
            continue
        probs_top = float(r["ml_top_prob"]) if "ml_top_prob" in df.columns else 0.0
        v = evaluate_guard(r["llm_raw_action"], ml_class=str(r["ml_predicted_class"]), ml_prob=probs_top,
                           neighbors=r["neighbors"], ioc_hits=int(r.get("ioc_hits", 0) or 0), cfg=cfg)
        out.append(v.action)
    return pd.Series(out, index=df.index)


def score(true: pd.Series, pred: pd.Series) -> Dict[str, float]:
    n = len(true)
    decided = pred != ABSTAIN
    correct = (pred == true) & decided
    bad = true != "VALID"
    good = true == "VALID"
    return {
        "coverage": float(decided.mean()),
        "resolved_acc": float(correct[decided].mean()) if decided.any() else float("nan"),
        "e2e_acc": float(correct.mean()),
        "leak_rate": float(((pred == "VALID") & bad).sum() / max(1, bad.sum())),
        "false_block_rate": float((decided & (pred != "VALID") & good).sum() / max(1, good.sum())),
        "valid_quarantined_rate": float(((pred == ABSTAIN) & good).sum() / max(1, good.sum())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--max-leak", type=float, default=0.02,
                    help="max share of spam/phishing allowed to be auto-ALLOWED (default 2%%)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="evals/results/generated/guard_sweep.csv")
    ap.add_argument("--agreement-review", action="store_true",
                    help="replay the current agreement-review policy on a complete disabled run")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if a.agreement_review:
        pred = replay_agreement_review(df)
        print(f"Agreement-review counterfactual on {len(df)} historical rows (not a new model run):")
        print({k: round(v, 4) for k, v in score(df["true_label"], pred).items()})
        print("The prior LLM saw the ML prediction; this replay is not independent model agreement.")
        return
    df = _prep(df)
    if df.empty:
        raise SystemExit("No LLM-path rows with llm_raw_action in this CSV (fast path covered everything?).")

    base = score(df["true_label"], df["agent_action"])
    print(f"Rows on LLM path: {len(df)}")
    print("Current guard (as logged):", {k: round(v, 4) for k, v in base.items()})
    off = simulate(df, GuardConfig(enable_ml_veto=False, enable_neighbor_veto=False))
    print("No guard at all          :", {k: round(v, 4) for k, v in score(df["true_label"], off).items()})

    rows: List[Dict[str, Any]] = []
    grid = itertools.product(
        [False, True], [0.5, 0.6, 0.7, 0.8, 0.9],                     # ml veto on/off, min prob
        [False, True], [0.6, 0.7, 0.75, 0.8, 0.85, 0.9], [0.5, 0.6, 0.8], [1, 2, 3],  # neighbour veto
    )
    seen = set()
    for ml_on, ml_p, nb_on, nb_sim, nb_frac, nb_n in grid:
        key = (ml_on, ml_p if ml_on else None, nb_on, (nb_sim, nb_frac, nb_n) if nb_on else None)
        if key in seen:
            continue
        seen.add(key)
        cfg = GuardConfig(enable_ml_veto=ml_on, ml_veto_min_prob=ml_p, enable_neighbor_veto=nb_on,
                          neighbor_min_sim=nb_sim, neighbor_min_nonbenign_frac=nb_frac, neighbor_min_count=nb_n)
        s = score(df["true_label"], simulate(df, cfg))
        rows.append({"ml_veto": ml_on, "ml_min_prob": ml_p if ml_on else None,
                     "nb_veto": nb_on, "nb_min_sim": nb_sim if nb_on else None,
                     "nb_min_frac": nb_frac if nb_on else None, "nb_min_count": nb_n if nb_on else None, **s})

    res = pd.DataFrame(rows)
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(output, index=False)
    ok = res[res["leak_rate"] <= a.max_leak].sort_values(["e2e_acc", "leak_rate", "valid_quarantined_rate"],
                                                                 ascending=[False, True, True])
    print(f"\nTop {a.top} configs with leak_rate <= {a.max_leak:.0%} (by end-to-end accuracy):")
    print(ok.head(a.top).round(4).to_string(index=False) if not ok.empty
          else "None met the leak constraint; relax --max-leak or improve the LLM prompt.")
    print(f"\nFull grid written to {a.out}")
    print("Reminder: confirm the chosen thresholds on a DIFFERENT fold before reporting them.")


if __name__ == "__main__":
    main()
