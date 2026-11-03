"""Tune the policy guard offline, without re-running the LLM.

The eval CSV stores the LLM's raw verdict, the ML prediction and the retrieved neighbours, so the
guard (a pure function) can be replayed over a grid of thresholds.

    python -m evals.sweep_guard evals/live_audit_report_results_fastpath-enabled_*.csv --max-leak 0.02

IMPORTANT: tune on one fold/seed and REPORT on another (`run_evals.py --fold 1`, re-run populate with
the same --fold). Choosing thresholds on the very rows you then report is optimistic.
"""
import argparse
import itertools
import json
from typing import Any, Dict, List

import pandas as pd

from src.guard import GuardConfig, evaluate_guard

ABSTAIN = "QUARANTINE"


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
    ap.add_argument("--out", default="evals/guard_sweep.csv")
    a = ap.parse_args()

    df = _prep(pd.read_csv(a.csv))
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
    res.to_csv(a.out, index=False)
    ok = res[res["leak_rate"] <= a.max_leak].sort_values(["e2e_acc", "leak_rate", "valid_quarantined_rate"],
                                                                 ascending=[False, True, True])
    print(f"\nTop {a.top} configs with leak_rate <= {a.max_leak:.0%} (by end-to-end accuracy):")
    print(ok.head(a.top).round(4).to_string(index=False) if not ok.empty
          else "None met the leak constraint; relax --max-leak or improve the LLM prompt.")
    print(f"\nFull grid written to {a.out}")
    print("Reminder: confirm the chosen thresholds on a DIFFERENT fold before reporting them.")


if __name__ == "__main__":
    main()
