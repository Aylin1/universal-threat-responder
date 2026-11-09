import pandas as pd

from scripts import update_readme_results as report


def test_latest_reports_selects_each_mode_by_run_id(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    folder = tmp_path / "evals"
    folder.mkdir()
    for name in (
        "live_audit_report_results_fastpath-enabled_20260101_120000_000001.csv",
        "live_audit_report_results_fastpath-enabled_20260201_120000_000001.csv",
        "live_audit_report_results_fastpath-disabled_20260101_120000_000001.csv",
        "live_audit_report_results_agreement-review_20260301_120000_000001.csv",
        "live_audit_report_results_20260401_120000_000001.csv",
    ):
        (folder / name).touch()
    latest = report.latest_reports()
    assert latest["fastpath-enabled"][0] == "20260201_120000_000001"
    assert set(latest) == {"fastpath-enabled", "fastpath-disabled", "agreement-review"}


def test_render_reports_counts_abstentions_and_leaks(tmp_path):
    csv = tmp_path / "test.csv"
    pd.DataFrame([
        {"true_label": "SPAM", "agent_action": "VALID"},
        {"true_label": "VALID", "agent_action": "QUARANTINE"},
        {"true_label": "PHISHING", "agent_action": "PHISHING"},
    ]).to_csv(csv, index=False)
    text = report.render({"agreement-review": ("20260101_120000_000001", csv)})
    assert "| agreement-review | 20260101_120000_000001 | 3 | 66.7% | 50.0% | 33.3% | 50.0% | 0.0% |" in text
    assert "| fastpath-enabled | Not run |" in text


def test_latest_reports_finds_run_folders_and_legacy_files(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    evals = tmp_path / "evals"
    run = evals / "runs" / "agreement-review_20260301_120000_000001"
    run.mkdir(parents=True)
    legacy = evals / "live_audit_report_results_agreement-review_20260201_120000_000001.csv"
    legacy.touch()
    current = run / "live_audit_report_results_agreement-review_20260301_120000_000001.csv"
    current.touch()
    latest = report.latest_reports()
    assert latest["agreement-review"][1] == current


def test_latest_reports_finds_collected_historical_and_new_run(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "ROOT", tmp_path)
    results = tmp_path / "evals" / "results"
    historic = results / "historical"
    run = results / "runs" / "fastpath-enabled_20260301_120000_000001"
    historic.mkdir(parents=True)
    run.mkdir(parents=True)
    old = historic / "live_audit_report_results_fastpath-enabled_20260201_120000_000001.csv"
    new = run / "live_audit_report_results_fastpath-enabled_20260301_120000_000001.csv"
    disabled = historic / "live_audit_report_results_fastpath-disabled_20260201_120000_000001.csv"
    for path in (old, new, disabled):
        path.touch()
    latest = report.latest_reports()
    assert latest["fastpath-enabled"][1] == new
    assert latest["fastpath-disabled"][1] == disabled
