"""Refresh benchmark_results.md from the newest run of each evaluation mode."""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DOC = ROOT / "benchmark_results.md"
NAME_RE = re.compile(
    r"live_audit_report_results_(fastpath-enabled|fastpath-disabled|agreement-review)_(\d{8}_\d{6}_\d+)$"
)


def latest_reports():
    latest = {}
    evals = ROOT / "evals"
    paths = list(evals.glob("live_audit_report_results_*.csv"))
    paths.extend((evals / "runs").glob("*/live_audit_report_results_*.csv"))
    paths.extend((evals / "results" / "historical").glob("live_audit_report_results_*.csv"))
    paths.extend((evals / "results" / "runs").glob("*/live_audit_report_results_*.csv"))
    for path in paths:
        match = NAME_RE.fullmatch(path.stem)
        if match and (match.group(1) not in latest or match.group(2) > latest[match.group(1)][0]):
            latest[match.group(1)] = (match.group(2), path)
    return latest


def render(reports):
    lines = [
        "# Benchmark results",
        "",
        "Latest timestamped runs per mode. Historical fast-path runs use the old LLM/guard policy; "
        "they do not measure the current agreement-review policy.",
        "",
        "| Mode | Run | Samples | Coverage | Resolved accuracy | End-to-end accuracy | Leak rate | False-block rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("fastpath-enabled", "fastpath-disabled", "agreement-review"):
        if mode not in reports:
            lines.append(f"| {mode} | Not run | — | — | — | — | — | — |")
            continue
        run_id, path = reports[mode]
        df = pd.read_csv(path)
        true, pred = df["true_label"], df["agent_action"]
        decided = pred != "QUARANTINE"
        correct = pred == true
        bad, good = true != "VALID", true == "VALID"
        pct = lambda x: f"{100 * x:.1f}%"
        resolved = pct(correct[decided].mean()) if decided.any() else "n/a"
        leak = pct(((pred == "VALID") & bad).sum() / bad.sum()) if bad.any() else "n/a"
        false_block = pct((decided & (pred != "VALID") & good).sum() / good.sum()) if good.any() else "n/a"
        report = path.with_suffix(".md")
        link = f"[{run_id}]({report.as_posix().replace(ROOT.as_posix() + '/', '')})" if report.exists() else run_id
        lines.append(f"| {mode} | {link} | {len(df):,} | {pct(decided.mean())} | {resolved} | "
                     f"{pct(correct.mean())} | {leak} | {false_block} |")
    lines.extend([
        "",
        "Coverage is the share not quarantined. Resolved accuracy is measured only on those decisions; "
        "end-to-end accuracy counts quarantine as incorrect. Leak rate is the share of non-benign mail "
        "automatically marked VALID; false-block rate is the share of valid mail classified SPAM or PHISHING.",
        "",
        "Refresh with `python -m scripts.update_readme_results`. Each linked report contains its latency "
        "and data-integrity measurements. Runs may not be directly comparable if their dataset, split, "
        "model, feeds, or policy differs.",
        "",
    ])
    return "\n".join(lines)


def main():
    reports = latest_reports()
    if not reports:
        raise SystemExit("No timestamped evaluation reports found in evals/.")
    RESULTS_DOC.write_text(render(reports), encoding="utf-8")
    for mode, (_, path) in sorted(reports.items()):
        print(f"{mode}: {path.name}")
    print(f"{RESULTS_DOC.name} updated.")


if __name__ == "__main__":
    main()
