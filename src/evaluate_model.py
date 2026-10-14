"""Evaluate the selected XGBoost model once on the held-out test split.

This script does not train a model, select a model, or tune a threshold. It
uses the estimator and validation-selected threshold saved during Phase 4.
The held-out test set is part of the fraud-enriched development sample, whose
fraud prevalence differs from the original population prevalence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = PROJECT_ROOT / "data" / "processed" / "prepared"
MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost.joblib"
TEST_PATH = PREPARED_DIR / "test.joblib"
PREPROCESSOR_PATH = PREPARED_DIR / "preprocessing.joblib"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "test_evaluation_report.md"
EXPECTED_TEST_ROWS = 7_381
EXPECTED_TEST_FRAUD = 960
EXPECTED_TEST_LEGITIMATE = 6_421
EXPECTED_FEATURE_COUNT = 39
TEST_STEP_RANGE = (156, 179)


def _check_required_paths() -> None:
    for path in (MODEL_PATH, TEST_PATH, PREPROCESSOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact not found: {path}")


def _load_model_artifact() -> dict[str, Any]:
    artifact = joblib.load(MODEL_PATH)
    required_keys = {
        "estimator",
        "selected_validation_threshold",
        "feature_names",
        "configuration",
        "preprocessing_artifact",
    }
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError(
            f"Model artifact must be a dictionary containing {sorted(required_keys)}"
        )
    estimator = artifact["estimator"]
    if not isinstance(estimator, XGBClassifier):
        raise TypeError(
            "Saved estimator must be an xgboost.XGBClassifier, "
            f"got {type(estimator).__module__}.{type(estimator).__name__}"
        )
    threshold = artifact["selected_validation_threshold"]
    if not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("Saved validation threshold must be numeric")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Saved validation threshold is outside [0, 1]: {threshold}")
    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Model artifact must contain exactly "
            f"{EXPECTED_FEATURE_COUNT} transformed feature names"
        )
    configuration = artifact["configuration"]
    if (
        not isinstance(configuration, dict)
        or configuration.get("class") != "xgboost.XGBClassifier"
    ):
        raise ValueError(
            "Model artifact configuration must identify xgboost.XGBClassifier"
        )
    preprocessing_reference = artifact["preprocessing_artifact"]
    if not isinstance(preprocessing_reference, str):
        raise TypeError("Preprocessing artifact reference must be a path string")
    referenced_preprocessor = (PROJECT_ROOT / preprocessing_reference).resolve()
    if referenced_preprocessor != PREPROCESSOR_PATH.resolve():
        raise ValueError(
            "Model artifact references a different preprocessing artifact: "
            f"{preprocessing_reference}"
        )
    return artifact


def _validate_preprocessor(
    preprocessor: Any, expected_feature_names: list[str]
) -> None:
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise TypeError(
            "Preprocessing artifact cannot provide transformed feature names"
        )
    preprocessor_feature_names = preprocessor.get_feature_names_out().tolist()
    if preprocessor_feature_names != expected_feature_names:
        raise ValueError(
            "Saved preprocessing feature names do not match the model artifact"
        )


def _load_and_validate_test(
    expected_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = joblib.load(TEST_PATH)
    required_keys = {"X", "y", "feature_names", "step", "source_row_position"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(
            f"Test artifact must contain keys {sorted(required_keys)}"
        )

    features = bundle["X"]
    labels = np.asarray(bundle["y"])
    feature_names = list(bundle["feature_names"])
    steps = np.asarray(bundle["step"])
    row_positions = np.asarray(bundle["source_row_position"])

    if not hasattr(features, "shape") or len(features.shape) != 2:
        raise ValueError("Test feature matrix must be two-dimensional")
    if features.shape != (EXPECTED_TEST_ROWS, EXPECTED_FEATURE_COUNT):
        raise ValueError(
            f"Test feature matrix has shape {features.shape}; expected "
            f"({EXPECTED_TEST_ROWS}, {EXPECTED_FEATURE_COUNT})"
        )
    if feature_names != expected_feature_names:
        raise ValueError(
            "Test feature names or ordering do not match the saved model"
        )
    if labels.ndim != 1 or len(labels) != EXPECTED_TEST_ROWS:
        raise ValueError("Test labels are not row-aligned with the feature matrix")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Test labels must contain only binary values 0 and 1")
    if steps.ndim != 1 or len(steps) != EXPECTED_TEST_ROWS:
        raise ValueError("Test steps are not row-aligned with the feature matrix")
    if not np.all((steps >= TEST_STEP_RANGE[0]) & (steps <= TEST_STEP_RANGE[1])):
        raise ValueError(
            f"Test steps must all be within {TEST_STEP_RANGE[0]}-"
            f"{TEST_STEP_RANGE[1]}"
        )
    if not np.array_equal(
        np.unique(steps), np.arange(TEST_STEP_RANGE[0], TEST_STEP_RANGE[1] + 1)
    ):
        raise ValueError(
            f"Test artifact must cover every step from {TEST_STEP_RANGE[0]} "
            f"through {TEST_STEP_RANGE[1]}"
        )
    if row_positions.ndim != 1 or len(row_positions) != EXPECTED_TEST_ROWS:
        raise ValueError("Test source-row positions are not row-aligned")
    if len(np.unique(row_positions)) != EXPECTED_TEST_ROWS:
        raise ValueError("Test artifact repeats source-row positions")

    fraud_count = int(np.count_nonzero(labels == 1))
    legitimate_count = int(np.count_nonzero(labels == 0))
    if (fraud_count, legitimate_count) != (
        EXPECTED_TEST_FRAUD,
        EXPECTED_TEST_LEGITIMATE,
    ):
        raise ValueError(
            "Test class counts mismatch: "
            f"fraud={fraud_count}, legitimate={legitimate_count}; expected "
            f"fraud={EXPECTED_TEST_FRAUD}, legitimate={EXPECTED_TEST_LEGITIMATE}"
        )
    return features, labels, steps


def _render_report(
    metrics: dict[str, Any],
    row_count: int,
    fraud_count: int,
    predicted_fraud_count: int,
    fraud_prevalence: float,
    minimum_step: int,
    maximum_step: int,
    threshold: float,
) -> str:
    matrix = metrics["confusion_matrix"]
    lines = [
        "# Held-Out Test Evaluation",
        "",
        "## Evaluation scope",
        "",
        "These are final held-out test results for the previously selected "
        "XGBoost model. The saved estimator was used without retraining, and "
        "the threshold saved from validation was applied without test-based "
        "tuning or model selection.",
        "",
        "The test set is drawn from a fraud-enriched development sample. Its "
        "observed prevalence should not be interpreted as the original "
        "population prevalence (1.210811%).",
        "",
        "## Test set summary",
        "",
        f"- Rows: {row_count:,}",
        f"- Step range: {minimum_step}-{maximum_step}",
        f"- Actual fraud cases: {fraud_count:,}",
        f"- Predicted fraud cases: {predicted_fraud_count:,}",
        f"- Fraud prevalence: {fraud_prevalence:.6f}%",
        f"- Applied validation-selected threshold: {threshold:.8f}",
        "",
        "## Held-out test metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Precision | {metrics['precision']:.6f} |",
        f"| Recall | {metrics['recall']:.6f} |",
        f"| F1-score | {metrics['f1']:.6f} |",
        f"| Average Precision (AP) | {metrics['average_precision_ap']:.6f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.6f} |",
        "",
        "Average Precision (AP) is computed with scikit-learn's "
        "`average_precision_score`; it is not described as trapezoidal "
        "precision-recall area.",
        "",
        "Confusion matrix uses rows=true labels `[legitimate, fraud]` and "
        "columns=predicted labels `[legitimate, fraud]`:",
        "",
        f"- Legitimate row: `{matrix[0]}`",
        f"- Fraud row: `{matrix[1]}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    _check_required_paths()
    model_artifact = _load_model_artifact()
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = model_artifact["feature_names"]
    _validate_preprocessor(preprocessor, feature_names)

    estimator = model_artifact["estimator"]
    if getattr(estimator, "n_features_in_", None) != len(feature_names):
        raise ValueError(
            "Estimator input feature count does not match its stored feature names"
        )
    if not np.array_equal(np.asarray(estimator.classes_), np.array([0, 1])):
        raise ValueError("XGBoost estimator must expose classes [0, 1]")

    test_features, test_labels, test_steps = _load_and_validate_test(feature_names)
    threshold = float(model_artifact["selected_validation_threshold"])
    class_labels = np.asarray(estimator.classes_)
    positive_class_index = int(np.flatnonzero(class_labels == 1)[0])
    positive_scores = estimator.predict_proba(test_features)[:, positive_class_index]
    if len(positive_scores) != len(test_labels) or not np.isfinite(
        positive_scores
    ).all():
        raise ValueError("Estimator produced invalid or misaligned test probabilities")

    predicted_labels = positive_scores >= threshold
    fraud_count = int(np.count_nonzero(test_labels == 1))
    predicted_fraud_count = int(np.count_nonzero(predicted_labels))
    confusion = confusion_matrix(test_labels, predicted_labels, labels=[0, 1])
    metrics = {
        "precision": float(
            precision_score(test_labels, predicted_labels, zero_division=0)
        ),
        "recall": float(recall_score(test_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(test_labels, predicted_labels, zero_division=0)),
        "average_precision_ap": float(
            average_precision_score(test_labels, positive_scores)
        ),
        "roc_auc": float(roc_auc_score(test_labels, positive_scores)),
        "confusion_matrix": confusion.astype(int).tolist(),
    }
    fraud_prevalence = fraud_count / len(test_labels) * 100
    report = _render_report(
        metrics,
        len(test_labels),
        fraud_count,
        predicted_fraud_count,
        fraud_prevalence,
        int(test_steps.min()),
        int(test_steps.max()),
        threshold,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(
        f"Held-out test: rows={len(test_labels):,}, "
        f"actual_fraud={fraud_count:,}, "
        f"predicted_fraud={predicted_fraud_count:,}, "
        f"fraud_prevalence={fraud_prevalence:.6f}%"
    )
    print(
        f"Threshold={threshold:.8f}, "
        f"precision={metrics['precision']:.6f}, "
        f"recall={metrics['recall']:.6f}, "
        f"F1={metrics['f1']:.6f}, "
        f"AP={metrics['average_precision_ap']:.6f}, "
        f"ROC-AUC={metrics['roc_auc']:.6f}"
    )
    print(f"Held-out test report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
