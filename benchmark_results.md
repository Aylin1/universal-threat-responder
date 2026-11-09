# Benchmark results

Latest timestamped runs per mode. Historical fast-path runs use the old LLM/guard policy; they do not measure the current agreement-review policy.

| Mode | Run | Samples | Coverage | Resolved accuracy | End-to-end accuracy | Leak rate | False-block rate |
|---|---|---:|---:|---:|---:|---:|---:|
| fastpath-enabled | [20261102_215446_001853](evals/results/historical/live_audit_report_results_fastpath-enabled_20261102_215446_001853.md) | 3,960 | 94.4% | 93.0% | 87.8% | 2.8% | 8.6% |
| fastpath-disabled | [20261102_222648_585601](evals/results/historical/live_audit_report_results_fastpath-disabled_20261102_222648_585601.md) | 3,960 | 83.7% | 94.0% | 78.7% | 1.3% | 6.4% |
| agreement-review | Not run | — | — | — | — | — | — |

Coverage is the share not quarantined. Resolved accuracy is measured only on those decisions; end-to-end accuracy counts quarantine as incorrect. Leak rate is the share of non-benign mail automatically marked VALID; false-block rate is the share of valid mail classified SPAM or PHISHING.

Refresh with `python -m scripts.update_readme_results`. Each linked report contains its latency and data-integrity measurements. Runs may not be directly comparable if their dataset, split, model, feeds, or policy differs.
