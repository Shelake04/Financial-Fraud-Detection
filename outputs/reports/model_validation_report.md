# Candidate Model Validation Results

## Scope and prevalence

All candidate metrics and decision thresholds in this report use the validation split only. The test artifact was not loaded or evaluated.

The development sample has **14.4% fraud prevalence**. The test partition has approximately **13.01%** fraud prevalence, compared with **1.210811%** in the original population. Validation metrics describe the enriched development sample, not the original population.

## Model selection and threshold

Selected candidate by highest validation Average Precision (AP): **XGBoost**.
Each model threshold maximizes validation F1; if multiple thresholds tie, the largest tied threshold is selected. No threshold or candidate selection used test labels or scores.

Confusion matrices use rows=true labels `[legitimate, fraud]` and columns=predicted labels `[legitimate, fraud]`.

## Validation metrics

| Model | Threshold | Precision | Recall | F1 | Average Precision (AP) | ROC-AUC | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Logistic Regression | 0.67690998 | 0.941929 | 0.957000 | 0.949405 | 0.980507 | 0.997712 | `[6513, 59] / [43, 957]` |
| Random Forest | 0.47333333 | 0.945152 | 0.965000 | 0.954973 | 0.981798 | 0.996749 | `[6516, 56] / [35, 965]` |
| XGBoost | 0.71875411 | 0.939072 | 0.971000 | 0.954769 | 0.990702 | 0.998618 | `[6509, 63] / [29, 971]` |

## Reproducibility and configurations

Random seed: `42`.

### Logistic Regression

```json
{
  "class": "sklearn.linear_model.LogisticRegression",
  "class_weight": "balanced",
  "max_iter": 2000,
  "random_state": 42,
  "solver": "liblinear"
}
```

Model artifact: `outputs\models\logistic_regression.joblib`.
Associated preprocessing artifact: `data\processed\prepared\preprocessing.joblib`.
Transformed feature names are stored alongside the estimator.

### Random Forest

```json
{
  "class": "sklearn.ensemble.RandomForestClassifier",
  "class_weight": "balanced_subsample",
  "max_features": "sqrt",
  "n_estimators": 300,
  "n_jobs": -1,
  "random_state": 42
}
```

Model artifact: `outputs\models\random_forest.joblib`.
Associated preprocessing artifact: `data\processed\prepared\preprocessing.joblib`.
Transformed feature names are stored alongside the estimator.

### XGBoost

```json
{
  "class": "xgboost.XGBClassifier",
  "colsample_bytree": 0.8,
  "eval_metric": "logloss",
  "learning_rate": 0.05,
  "max_depth": 5,
  "n_estimators": 300,
  "n_jobs": -1,
  "objective": "binary:logistic",
  "random_state": 42,
  "subsample": 0.8,
  "tree_method": "hist",
  "scale_pos_weight": 5.688358778625954
}
```

Model artifact: `outputs\models\xgboost.joblib`.
Associated preprocessing artifact: `data\processed\prepared\preprocessing.joblib`.
Transformed feature names are stored alongside the estimator.
