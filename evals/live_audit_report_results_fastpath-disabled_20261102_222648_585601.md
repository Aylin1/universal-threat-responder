# Evaluation 20261102_222648_585601 (fastpath-disabled, alpha=0.05, fold=0)

Samples: 3960

## System comparison

`End-to-end` counts abstentions (QUARANTINE) as wrong, so it is comparable across systems. `Resolved` is accuracy only on emails the system decided. 95% bootstrap CIs in brackets.

| System | Coverage | Resolved acc. | End-to-end acc. | Leak rate | False-block rate |
|---|---|---|---|---|---|
| ML only (argmax, no abstention) | 100.0% | 90.6% [89.6, 91.5] | 90.6% [89.6, 91.5] | 4.6% | 13.4% |
| ML + conformal (abstain unless singleton set) | 87.8% | 94.7% [93.9, 95.4] | 83.1% [81.9, 84.2] | 2.2% | 6.9% |
| FULL: conformal fast-path + RAG/LLM + guard | 83.7% | 94.0% [93.2, 94.8] | 78.7% [77.4, 79.9] | 1.3% | 6.4% |
| FULL without the policy guard (counterfactual) | 99.3% | 83.8% [82.6, 84.9] | 83.2% [82.0, 84.4] | 18.0% | 6.4% |

Leak rate = spam/phishing auto-ALLOWED / all spam+phishing. False-block rate = valid mail auto-blocked or spam-filtered / all valid mail.

## By route

| Route | Share | Coverage | Resolved acc. | End-to-end acc. |
|---|---|---|---|---|
| ESCALATION_RAG_LLM | 100.0% | 83.7% | 94.0% | 78.7% |

## Confusion matrix (full system; rows = truth)

| truth \ pred | VALID | SPAM | PHISHING | QUARANTINE |
|---|---|---|---|---|
| VALID | 1043 | 58 | 26 | 193 |
| SPAM | 22 | 827 | 64 | 377 |
| PHISHING | 12 | 17 | 1245 | 76 |

## Per-class (decided emails only)

| Class | Precision | Recall (decided) | Support |
|---|---|---|---|
| VALID | 96.8% | 92.5% | 1127 |
| SPAM | 91.7% | 90.6% | 913 |
| PHISHING | 93.3% | 97.7% | 1274 |

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
- `llm_verbalized_ece`: 0.0957
- `llm_verbalized_brier`: 0.1424
- `llm_raw_accuracy`: 0.8323
- `llm_n`: 3960

## Policy-guard impact (rows where the LLM said VALID)

- `llm_said_valid`: 1700
- `vetoed`: 619
- `veto_rate`: 0.364
- `vetoed_but_actually_valid`: 179
- `vetoed_and_actually_bad`: 440
- `unvetoed_but_actually_bad`: 36

## Latency (ms, per email)

- retrieval (embedding + Milvus): mean 160.7
- graph (LLM or bypass node only): mean 1732.0
- end-to-end: mean 1892.6, p95 2555.8

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
