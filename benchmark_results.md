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
| **Accuracy** | 90.02% | 85.83% |
| **Macro Precision** | 90.29% | 85.80% |
| **Macro Recall** | 88.22% | 78.88% |
| **Macro F1** | 0.884 | 0.789 |
| **Quarantined Rate** | 33.74% | 45.66% |
| **Mean ML Entropy** | 0.4310 bits | 0.4310 bits |
| **Mean RAG Top-1 Similarity** | 0.5821 | 0.5807 |
| **Mean LLM Calibration Error** | 0.0369 | 0.0383 |

Latest runs: fast-path enabled = `20261016_144758_055131`, fast-path disabled = `20261016_155121_468311`. Metrics are computed on resolved (non-quarantined) rows. Refresh with `python -m scripts.update_readme_results`.

High-risk threat indicator hits (latest enabled run): **10,170**
<!-- PERF:END -->

## Interpreting the Results

- **Metrics are computed on resolved rows only.** Quarantined samples are an abstention, so they are excluded from accuracy, precision, recall, and F1 and reported separately as the *Quarantined Rate*. Accuracy is therefore conditional on the system choosing to decide.
- **Fast-path routing helps on both axes.** With it enabled, 41.44% of samples (1,641) bypass the LLM/RAG pipeline. Those are answered with ~99.5% accuracy, which lifts overall accuracy from 85.83% to 90.02% and macro F1 from 0.789 to 0.884. It also reduces quarantines from 45.66% to 33.74%.
- **The LLM/RAG path is still the weak point.** Among resolved rows, the escalated path reaches ~74.2% accuracy (983 samples) with fast-path enabled and ~85.8% (2,152 samples) when every sample takes it. The fast path keeps the easy samples, so these two figures are not directly comparable. Full-path macro recall is low (78.88%) because of the high abstention rate (45.66% quarantined).
- **Dominant error: SPAM predicted as VALID.** In the fast-path-enabled run, 154 of 714 resolved SPAM samples (22%) were labelled VALID, versus only 1 SPAM→PHISHING. PHISHING is the strongest class (1,163 of 1,186 resolved correct), and false VALID verdicts on PHISHING are rare (22).
- **Calibration and retrieval.** Mean LLM calibration error is low (0.037 vs 0.038). Mean RAG top-1 similarity is ~0.58 in both modes. Earlier runs reported ~0.20 because Milvus returns squared L2 distance and the code computed `1 - distance`; the adapter now converts with `cos = 1 - d / 2` (see `l2_to_cosine_similarity`). ML entropy is identical (0.4310 bits) because it is computed before routing.
- **Retrieval-augmented prompt.** The LLM prompt now includes the top-3 retrieved threat-intelligence matches and the SOP policies. Compared with the previous runs (same dataset), fast-path-enabled accuracy rose from 88.56% to 90.02% (F1 0.866 → 0.884) and quarantines fell from 37.10% to 33.74%. Full-path accuracy rose from 81.08% to 85.83% (F1 0.751 → 0.789), and quarantines fell from 48.21% to 45.66%. These are single runs with a stochastic local LLM, so small differences should not be over-interpreted.

### Confusion matrix, resolved rows (rows = true label, columns = agent action)

Fast-path enabled:

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,163 | 1 | 22 |
| **SPAM** | 52 | 508 | 154 |
| **VALID** | 17 | 16 | 691 |

Fast-path disabled:

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,107 | 2 | 23 |
| **SPAM** | 49 | 197 | 201 |
| **VALID** | 15 | 15 | 543 |

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
