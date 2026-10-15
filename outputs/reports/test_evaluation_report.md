# Held-Out Test Evaluation

## Evaluation scope

These are final held-out test results for the previously selected XGBoost model. The saved estimator was used without retraining, and the threshold saved from validation was applied without test-based tuning or model selection.

The test set is drawn from a fraud-enriched development sample. Its observed prevalence should not be interpreted as the original population prevalence (1.210811%).

## Test set summary

- Rows: 7,381
- Step range: 156-179
- Actual fraud cases: 960
- Predicted fraud cases: 989
- Fraud prevalence: 13.006368%
- Applied validation-selected threshold: 0.71875411

## Held-out test metrics

| Metric | Value |
| --- | ---: |
| Precision | 0.938322 |
| Recall | 0.966667 |
| F1-score | 0.952283 |
| Average Precision (AP) | 0.992662 |
| ROC-AUC | 0.998882 |

Average Precision (AP) is computed with scikit-learn's `average_precision_score`; it is not described as trapezoidal precision-recall area.

Confusion matrix uses rows=true labels `[legitimate, fraud]` and columns=predicted labels `[legitimate, fraud]`:

- Legitimate row: `[6360, 61]`
- Fraud row: `[32, 928]`
