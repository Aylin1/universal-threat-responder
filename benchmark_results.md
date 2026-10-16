# Benchmark & Telemetry Results

Back to the [project README](README.md).

The latest evaluation was run on **3,960 samples** from:

`data/spam/PhishFuzzer_emails_entity_rephrased_v1.json`

Threat intelligence was synchronized before evaluation, including **30 OpenPhish records**, **30 Abuse.ch URLhaus records**, and **65 email threat records**.

## Performance Summary

<!-- PERF:START -->
| Metric | Fast-Path Enabled | Fast-Path Disabled |
| :--- | :---: | :---: |
| **Samples** | 3,960 | 3,960 |
| **Fast-Path Bypass** | 41.44% | 0.00% |
| **Accuracy** | 88.56% | 81.08% |
| **Macro Precision** | 87.50% | 79.00% |
| **Macro Recall** | 87.48% | 78.66% |
| **Macro F1** | 0.866 | 0.751 |
| **Quarantined Rate** | 37.10% | 48.21% |
| **Mean ML Entropy** | 0.4310 bits | 0.4310 bits |
| **Mean RAG Top-1 Similarity** | 0.2046 | 0.2037 |
| **Mean LLM Calibration Error** | 0.0449 | 0.0476 |

Latest runs: fast-path enabled = `20261015_144005_794922`, fast-path disabled = `20261015_155558_185194`. Metrics are computed on resolved (non-quarantined) rows. Refresh with `python -m scripts.update_readme_results`.

High-risk threat indicator hits (latest enabled run): **13,260**
<!-- PERF:END -->

## Interpreting the Results

- **Metrics are computed on resolved rows only.** Quarantined samples are an abstention, so they are excluded from accuracy, precision, recall, and F1 and reported separately as the *Quarantined Rate*. Accuracy is therefore conditional on the system choosing to decide.
- **Fast-path routing helps on both axes.** With it enabled, 41.44% of samples (1,641) bypass the LLM/RAG pipeline. Those are answered with ~99.5% accuracy, which lifts overall accuracy from 81.08% to 88.56% and macro F1 from 0.751 to 0.866. It also reduces quarantines from 48.21% to 37.10%.
- **The LLM/RAG path is the weak point.** Among resolved rows, the escalated path reaches ~67.4% accuracy with fast-path enabled and ~81.1% when every sample takes it. The samples the fast path keeps are the easy ones, so the two figures are not directly comparable.
- **Dominant error: SPAM predicted as VALID.** In the fast-path-enabled run, 180 of 777 resolved SPAM samples (23%) were labelled VALID, versus only 9 SPAM→PHISHING. PHISHING is the strongest class (1,120 of 1,145 resolved correct), and false VALID verdicts on PHISHING are rare (16).
- **Calibration and retrieval.** Mean LLM calibration error is low (0.045 vs 0.048). Mean RAG top-1 similarity is low (~0.20) in both modes, so retrieval grounding is weak. ML entropy is identical (0.4310 bits) because it is computed before routing.

### Confusion matrix, resolved rows (rows = true label, columns = agent action)

Fast-path enabled:

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,120 | 9 | 16 |
| **SPAM** | 39 | 558 | 180 |
| **VALID** | 16 | 25 | 528 |

Fast-path disabled:

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,030 | 9 | 24 |
| **SPAM** | 38 | 310 | 280 |
| **VALID** | 12 | 25 | 323 |

## Live Telemetry

| Metric                              |  Latest Value  |
| :---------------------------------- | :------------: |
| **OpenPhish Records**               |     **30**     |
| **URLhaus Records**                 |     **30**     |
| **Email Threat Records**            |     **65**     |
| **Conformal Bounds**                | `[0.05, 0.95]` |

> The fast-path and full-path results use the same 3,960-sample dataset. This is an observational configuration comparison rather than a controlled causal experiment, since enabling fast-path routing changes which samples reach the LLM/RAG pipeline.

## Reproducing

```bash
python -m evals.run_evals                      # fast-path enabled
python -m evals.run_evals --disable-fast-path  # full LLM/RAG path
python -m scripts.update_readme_results        # refresh the Performance Summary above
```

Each run writes `evals/live_audit_report_results_<fastpath-enabled|fastpath-disabled>_<run_id>.csv`.

## Data Provenance

- **[PhishFuzzer Dataset](https://github.com/DataPhish/PhishFuzzer)** — phishing, spam, and legitimate enterprise email samples.
- **[OpenPhish](https://openphish.com/)** — live phishing intelligence.
- **[Abuse.ch URLhaus](https://urlhaus.abuse.ch/)** — malicious URL and malware-distribution intelligence.
- Behavioral threat patterns and SOP-based indicators are also used for retrieval and enrichment.

Benchmark results are dataset- and configuration-specific and should not be interpreted as production security performance.
