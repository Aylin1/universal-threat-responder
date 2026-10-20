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
| **Fast-Path Bypass** | 42.42% | 0.00% |
| **Accuracy** | 94.47% | 91.15% |
| **Macro Precision** | 94.76% | 87.39% |
| **Macro Recall** | 92.85% | 82.64% |
| **Macro F1** | 0.937 | 0.843 |
| **Quarantined Rate** | 42.95% | 58.91% |
| **Mean ML Entropy** | 0.4298 bits | 0.4298 bits |
| **Mean RAG Top-1 Similarity** | 0.6192 | 0.6202 |
| **Mean LLM Calibration Error** | 0.0363 | 0.0378 |

Latest runs: fast-path enabled = `20261020_021654_177543`, fast-path disabled = `20261020_082153_967037`. Metrics are computed on resolved (non-quarantined) rows. Refresh with `python -m scripts.update_readme_results`.

High-risk threat indicator hits (latest enabled run): **16,650**
<!-- PERF:END -->

## Interpreting the Results

- **Metrics are computed on resolved rows only.** Quarantined samples are an abstention, so they are excluded from accuracy, precision, recall, and F1 and reported separately as the *Quarantined Rate*. Accuracy is therefore conditional on the system choosing to decide.
- **Fast-path routing helps on both axes.** With it enabled, 42.42% of samples (1,680) bypass the LLM/RAG pipeline. Those are answered with ~99.5% accuracy, which lifts accuracy on resolved rows from 91.15% to 94.47% and macro F1 from 0.843 to 0.937. It also reduces quarantines from 58.91% to 42.95%, and macro precision/recall rise from 87.39%/82.64% to 94.76%/92.85%.
- **Safer, but more conservative, than the previous benchmark.** Against the previous run (2026-10-16, same dataset), accuracy on resolved rows rose (90.02% → 94.47% enabled, 85.83% → 91.15% disabled) and so did macro F1 (0.884 → 0.937, 0.789 → 0.843). Threats (PHISHING or SPAM) wrongly labelled `VALID` fell sharply: 176 → 25 with fast-path enabled and 224 → 47 with it disabled. The cost is that the system abstains more: the quarantine rate rose from 33.74% to 42.95% (enabled) and from 45.66% to 58.91% (disabled). The share of **all** samples answered correctly fell from 59.65% to 53.89% (enabled) and from 46.64% to 37.45% (disabled). The new numbers are not strictly better: errors moved from wrong verdicts into abstentions.
- **Why quarantines rose.** Retrieval now embeds a condensed query (subject, up to three URLs, first 350 characters) instead of the whole email. Mean RAG top-1 similarity rose from ~0.58 to ~0.62, and the share of samples whose top match has `max_risk_score` ≥ 0.95 rose from ~75% to ~93%. The policy guard fires whenever that score is at least 0.95, so more LLM `VALID` verdicts are overridden (46.3% of samples above the bound end up quarantined with fast-path enabled, against 0.3% below it). The LLM confidence threshold (0.50) also now quarantines low-confidence verdicts, but the report CSV does not record which guard fired, so the split between the two was not measured.
- **The LLM/RAG path is still the weak point.** With fast-path enabled, the escalated path resolves only 579 of 2,280 samples (25.4%) and reaches ~80.0% accuracy on them. With every sample escalated it resolves 1,627 of 3,960 (41.1%) at ~91.2%. The fast path keeps the easy samples, so these two figures are not directly comparable, and the high abstention rate keeps full-path macro recall low (82.64%).
- **Dominant error is now abstention, not SPAM predicted as VALID.** Resolved SPAM labelled `VALID` fell from 154 (of 714) to 16 (of 579) with fast-path enabled. SPAM is instead the class most often quarantined (711 of 1,290), followed by `VALID` mail (829 of 1,320, 62.8%). The remaining resolved errors are mostly SPAM labelled PHISHING (59) and `VALID` mail labelled PHISHING or SPAM (36). PHISHING remains the strongest class (1,175 of 1,189 resolved correct).
- **Calibration and retrieval.** Mean LLM calibration error is low (0.036 vs 0.038). Mean RAG top-1 similarity is ~0.62 in both modes. Earlier runs reported ~0.20 because Milvus returns squared L2 distance and the code computed `1 - distance`; the adapter now converts with `cos = 1 - d / 2` (see `l2_to_cosine_similarity`). The retrieval-uncertainty flag now fires on 1.6% of samples (previously 7.3%) because the condensed query returns at least one relevant match for more emails. ML entropy is identical (0.4298 bits) because it is computed before routing.
- **Single runs.** Each figure comes from one run with a stochastic local LLM, so small differences should not be over-interpreted. The fast-path-enabled and -disabled runs were also executed hours apart (live threat feeds are re-synced before each run).

### Confusion matrix, resolved rows (rows = true label, columns = agent action)

Fast-path enabled (2,259 resolved, 1,701 quarantined):

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,175 | 5 | 9 |
| **SPAM** | 59 | 504 | 16 |
| **VALID** | 18 | 18 | 455 |

Fast-path disabled (1,627 resolved, 2,333 quarantined):

| True \ Predicted | PHISHING | SPAM | VALID |
| :--- | :---: | :---: | :---: |
| **PHISHING** | 1,122 | 5 | 10 |
| **SPAM** | 60 | 169 | 37 |
| **VALID** | 14 | 18 | 192 |

### Why the full path quarantines so often

Almost all full-path quarantines come from the **policy guard** in `src/core.py`, which converts an LLM `VALID` verdict into `QUARANTINE` when the retrieved `max_risk_score` is at least 0.95. In a re-run of 150 randomly chosen quarantined samples (previous benchmark), 149 were guard overrides; there were no parse failures or execution errors. In the latest full-path run, 83.0% of genuinely legitimate mail is quarantined, and 93% of all samples retrieve a match at or above 0.95, so the guard fires on most `VALID` verdicts. The live-feed and SOP records carry a fixed score (0.90-0.95) regardless of how close they are to the email, which is why the guard now requires corroborating evidence (see below).

**Guard change (pending re-benchmark).** The numbers above were measured before this change. The guard used to veto every LLM `VALID` verdict whenever any retrieved record scored at least 0.95, which is only topical overlap because threat-intel records carry a fixed score. It now vetoes `VALID` only with corroborating evidence: the email contains a known threat-intel URL, the high-risk match has cosine similarity of at least 0.80 (`cfg.risk.guard_similarity`), or the local ML model also predicted SPAM or PHISHING. Contexts without evidence fields still fail closed. Each report CSV now has a `guard_applied` column (`none`, `policy` or `low_confidence`) so the guards can be told apart. The expected effect is a lower quarantine rate with some of the previously quarantined threats now answered `VALID` when both the LLM and the ML model say so. Rerun both evaluation modes and `python -m scripts.update_readme_results` to measure it.

**Experiment: gating the guard on similarity alone (not adopted).** The guard was changed to veto `VALID` only when a high-risk match also had cosine similarity of at least 0.80 to the email (full-path run on the previous benchmark's code, same dataset; archived in `evals/experiments/`):

| Metric (full path, 2026-10-16 run) | Guard as shipped | Similarity-gated guard |
| :--- | :---: | :---: |
| Quarantined Rate | 45.66% | 0.08% |
| Resolved accuracy | 85.83% | 65.48% |
| Macro F1 | 0.789 | 0.600 |
| PHISHING emails labelled VALID | 23 (+218 quarantined) | 246 |
| SPAM emails labelled VALID | 201 (+843 quarantined) | 1,046 |

The quarantine rate collapsed, but only because the LLM labels most mail `VALID` (2,586 of 3,960) and the gate (mean top-1 similarity is only ~0.59) almost never fires. The guard was acting as a safety net against LLM over-prediction of `VALID`, so the change was reverted. The shipped fix instead also considers the ML prediction (see above); fixing the LLM's `VALID` bias, for example with spam/valid examples in the prompt, would help further.

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
