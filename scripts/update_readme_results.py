"""Refresh the README performance tables from the latest eval report CSVs.

Usage: python -m scripts.update_readme_results
"""
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPORT_GLOB = "live_audit_report_results_*.csv"
START, END = "<!-- PERF:START -->", "<!-- PERF:END -->"


NAME_RE = re.compile(r"live_audit_report_results_(?:fastpath-(enabled|disabled)_)?(.+)$")


def run_id(path: Path) -> str:
    return NAME_RE.match(path.stem).group(2)


def latest_reports():
    """Return the newest report per mode (run ids are timestamps, so they sort).

    The mode comes from the file name; legacy files without it are classified by content.
    """
    latest = {}
    for path in sorted((ROOT / "evals").glob(REPORT_GLOB), key=run_id):
        df = pd.read_csv(path)
        mode = NAME_RE.match(path.stem).group(1)
        if mode is None:
            mode = "enabled" if (df["route_taken"] == "FAST_PATH_BYPASS").any() else "disabled"
        latest[mode] = (path, df)
    return latest


def compute(df: pd.DataFrame) -> dict:
    quarantined = df["is_quarantined"].astype(bool)
    resolved = df[~quarantined]
    precision, recall, f1, _ = precision_recall_fscore_support(
        resolved["true_label"], resolved["agent_action"], average="macro", zero_division=0
    )
    return {
        "samples": f"{len(df):,}",
        "bypass": f"{(df['route_taken'] == 'FAST_PATH_BYPASS').mean() * 100:.2f}%",
        "acc": f"{accuracy_score(resolved['true_label'], resolved['agent_action']) * 100:.2f}%",
        "prec": f"{precision * 100:.2f}%",
        "rec": f"{recall * 100:.2f}%",
        "f1": f"{f1:.3f}",
        "quar": f"{quarantined.mean() * 100:.2f}%",
        "ent": f"{df['ml_shannon_entropy'].mean():.4f} bits",
        "rag": f"{df['rag_similarity'].mean():.4f}",
        "cal": f"{df['calibration_error'].mean():.4f}",
        "hits": int(df["high_risk_hits"].sum()),
    }


def render(reports: dict) -> str:
    cols = {m: compute(df) for m, (_, df) in reports.items()}
    na = {k: "n/a" for k in next(iter(cols.values()))}
    en, dis = cols.get("enabled", na), cols.get("disabled", na)
    rows = [
        ("Samples", "samples"), ("Fast-Path Bypass", "bypass"), ("Accuracy", "acc"),
        ("Macro Precision", "prec"), ("Macro Recall", "rec"), ("Macro F1", "f1"),
        ("Quarantined Rate", "quar"), ("Mean ML Entropy", "ent"),
        ("Mean RAG Top-1 Similarity", "rag"), ("Mean LLM Calibration Error", "cal"),
    ]
    lines = [
        "| Metric | Fast-Path Enabled | Fast-Path Disabled |",
        "| :--- | :---: | :---: |",
    ]
    lines += [f"| **{label}** | {en[k]} | {dis[k]} |" for label, k in rows]
    ids = ", ".join(
        f"fast-path {m} = `{run_id(reports[m][0])}`" for m in ("enabled", "disabled") if m in reports
    )
    lines += [
        "",
        f"Latest runs: {ids}. Metrics are computed on resolved (non-quarantined) rows. "
        "Refresh with `python -m scripts.update_readme_results`.",
        "",
        f"High-risk threat indicator hits (latest enabled run): **{en['hits']:,}**"
        if "enabled" in reports else "",
    ]
    return "\n".join(lines).rstrip()


def main() -> int:
    reports = latest_reports()
    if not reports:
        print("No eval reports found in evals/. Run `python -m evals.run_evals` first.")
        return 1
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"README is missing the {START} / {END} markers.")
        return 1
    block = f"{START}\n{render(reports)}\n{END}"
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    if "disabled" not in reports:
        print("Warning: no fast-path-disabled report found; column left as n/a.")
    README.write_text(text, encoding="utf-8", newline="\n")
    for mode, (path, _) in reports.items():
        print(f"{mode}: {path.name}")
    print("README.md updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
