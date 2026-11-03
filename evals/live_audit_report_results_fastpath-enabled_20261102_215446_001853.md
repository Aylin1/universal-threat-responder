# Evaluation 20261102_215446_001853 (fastpath-enabled, alpha=0.05, fold=0)

Samples: 3960

## System comparison

`End-to-end` counts abstentions (QUARANTINE) as wrong, so it is comparable across systems. `Resolved` is accuracy only on emails the system decided. 95% bootstrap CIs in brackets.

| System | Coverage | Resolved acc. | End-to-end acc. | Leak rate | False-block rate |
|---|---|---|---|---|---|
| ML only (argmax, no abstention) | 100.0% | 90.6% [89.6, 91.5] | 90.6% [89.6, 91.5] | 4.6% | 13.4% |
| ML + conformal (abstain unless singleton set) | 87.8% | 94.7% [93.9, 95.4] | 83.1% [81.9, 84.2] | 2.2% | 6.9% |
| FULL: conformal fast-path + RAG/LLM + guard | 94.4% | 93.0% [92.1, 93.8] | 87.8% [86.8, 88.8] | 2.8% | 8.6% |
| FULL without the policy guard (counterfactual) | 99.9% | 89.6% [88.7, 90.5] | 89.5% [88.6, 90.4] | 8.5% | 8.6% |

Leak rate = spam/phishing auto-ALLOWED / all spam+phishing. False-block rate = valid mail auto-blocked or spam-filtered / all valid mail.

## By route

| Route | Share | Coverage | Resolved acc. | End-to-end acc. |
|---|---|---|---|---|
| ESCALATION_RAG_LLM | 12.2% | 54.6% | 70.9% | 38.8% |
| FAST_PATH_BYPASS | 87.8% | 100.0% | 94.7% | 94.7% |

## Confusion matrix (full system; rows = truth)

| truth \ pred | VALID | SPAM | PHISHING | QUARANTINE |
|---|---|---|---|---|
| VALID | 1138 | 80 | 34 | 68 |
| SPAM | 48 | 1065 | 46 | 131 |
| PHISHING | 27 | 27 | 1275 | 21 |

## Per-class (decided emails only)

| Class | Precision | Recall (decided) | Support |
|---|---|---|---|
| VALID | 93.8% | 90.9% | 1252 |
| SPAM | 90.9% | 91.9% | 1159 |
| PHISHING | 94.1% | 95.9% | 1329 |

## ML-only coverage vs accuracy

| Min top prob | Coverage | Accuracy |
|---|---|---|
| 0.00 | 100.0% | 90.6% |
| 0.50 | 98.3% | 91.3% |
| 0.60 | 93.9% | 93.0% |
| 0.70 | 89.8% | 94.4% |
| 0.80 | 83.8% | 95.7% |
| 0.90 | 71.4% | 97.6% |
| 0.95 | 35.8% | 99.9% |

## Calibration (against correctness)

- `ml_ece`: 0.0260
- `ml_brier`: 0.0684
- `llm_verbalized_ece`: 0.3718
- `llm_verbalized_brier`: 0.3915
- `llm_raw_accuracy`: 0.5278
- `llm_n`: 485

## Policy-guard impact (rows where the LLM said VALID)

- `llm_said_valid`: 303
- `vetoed`: 216
- `veto_rate`: 0.713
- `vetoed_but_actually_valid`: 67
- `vetoed_and_actually_bad`: 149
- `unvetoed_but_actually_bad`: 20

## Latency (ms, per email)

- retrieval (embedding + Milvus): mean 141.8
- graph (LLM or bypass node only): mean 224.5
- end-to-end: mean 366.3, p95 1978.1

## Data integrity

```json
{
 "grouping": "Original_ID",
 "n_train": 15840,
 "n_test": 3960,
 "group_overlap": 0,
 "exact_duplicate_overlap": 0,
 "test_nn_cosine_mean": 0.4169412822920953,
 "test_nn_cosine_ge_0.90_rate": 0.0002525252525252525
}
```
