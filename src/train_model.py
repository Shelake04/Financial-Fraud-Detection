"""Train candidate classifiers and compare them using validation data only.

This script does not load or score the test split. The development sample has
14.4% fraud prevalence, and the held-out test partition has approximately
13.01%, compared with 1.210811% in the original population; reported validation
metrics therefore describe the fraud-enriched development sample.

Outputs:
- ``outputs/models/*.joblib``: candidate estimator, selected validation
  threshold, feature names, and the associated preprocessing artifact path.
- ``outputs/reports/model_validation_report.md``: validation-only metrics,
  thresholds, confusion matrices, and model configurations.

The selected model is the candidate with the highest validation Average
Precision (AP), as computed by scikit-learn's ``average_precision_score``.
AP is not trapezoidal area under a precision-recall curve. Each model's
decision threshold is selected on validation data by maximizing F1; ties
choose the largest threshold.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED_DIR = PROJECT_ROOT / "data" / "processed" / "prepared"
TRAIN_PATH = PREPARED_DIR / "train.joblib"
VALIDATION_PATH = PREPARED_DIR / "validation.joblib"
PREPROCESSOR_PATH = PREPARED_DIR / "preprocessing.joblib"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "model_validation_report.md"
RANDOM_SEED = 42
EXPECTED_SPLITS = {
    "train": {
        "rows": 35_047,
        "fraud": 5_240,
        "legitimate": 29_807,
        "first_step": 0,
        "last_step": 130,
    },
    "validation": {
        "rows": 7_572,
        "fraud": 1_000,
        "legitimate": 6_572,
        "first_step": 131,
        "last_step": 155,
    },
}
MODEL_CONFIGURATIONS = {
    "Logistic Regression": {
        "class": "sklearn.linear_model.LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": RANDOM_SEED,
        "solver": "liblinear",
    },
    "Random Forest": {
        "class": "sklearn.ensemble.RandomForestClassifier",
        "class_weight": "balanced_subsample",
        "max_features": "sqrt",
        "n_estimators": 300,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
    "XGBoost": {
        "class": "xgboost.XGBClassifier",
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 5,
        "n_estimators": 300,
        "n_jobs": -1,
        "objective": "binary:logistic",
        "random_state": RANDOM_SEED,
        "subsample": 0.8,
        "tree_method": "hist",
    },
}


def _check_artifact_paths() -> None:
    for artifact_path in (TRAIN_PATH, VALIDATION_PATH, PREPROCESSOR_PATH):
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Required prepared artifact not found: {artifact_path}")


def _require_xgboost() -> None:
    if importlib.util.find_spec("xgboost") is None:
        raise ModuleNotFoundError(
            "XGBoost is required to train all three candidate models but is not "
            "installed in the selected Python environment. It is listed in "
            "requirements.txt; install it in the project environment, then rerun. "
            "No candidate models have been trained."
        )


def _load_and_validate_bundle(
    bundle_path: Path,
    split_name: str,
    expected_feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    bundle = joblib.load(bundle_path)
    required_keys = {"X", "y", "feature_names", "source_row_position", "step"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(
            f"{split_name} artifact must contain keys {sorted(required_keys)}"
        )

    features = np.asarray(bundle["X"])
    labels = np.asarray(bundle["y"])
    row_positions = np.asarray(bundle["source_row_position"])
    steps = np.asarray(bundle["step"])
    feature_names = list(bundle["feature_names"])
    expected = EXPECTED_SPLITS[split_name]
    if features.ndim != 2 or features.shape[0] != expected["rows"]:
        raise ValueError(
            f"{split_name} feature matrix has invalid shape {features.shape}; "
            f"expected ({expected['rows']}, n_features)"
        )
    if labels.ndim != 1 or len(labels) != expected["rows"]:
        raise ValueError(f"{split_name} labels are not row-aligned")
    if row_positions.ndim != 1 or len(row_positions) != expected["rows"]:
        raise ValueError(f"{split_name} source positions are not row-aligned")
    if steps.ndim != 1 or len(steps) != expected["rows"]:
        raise ValueError(f"{split_name} steps are not row-aligned")
    if len(np.unique(row_positions)) != expected["rows"]:
        raise ValueError(f"{split_name} repeats source row positions")
    if len(feature_names) != features.shape[1]:
        raise ValueError(
            f"{split_name} has {features.shape[1]} columns but "
            f"{len(feature_names)} feature names"
        )
    if expected_feature_names is not None and feature_names != expected_feature_names:
        raise ValueError(
            f"{split_name} transformed feature names differ from training"
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{split_name} labels must contain only 0 and 1")

    counts = {
        "fraud": int(np.count_nonzero(labels == 1)),
        "legitimate": int(np.count_nonzero(labels == 0)),
    }
    if counts != {
        "fraud": expected["fraud"],
        "legitimate": expected["legitimate"],
    }:
        raise ValueError(
            f"{split_name} class counts mismatch: observed={counts}, "
            f"expected fraud={expected['fraud']}, "
            f"legitimate={expected['legitimate']}"
        )
    if not np.all(
        (steps >= expected["first_step"]) & (steps <= expected["last_step"])
    ):
        raise ValueError(
            f"{split_name} contains a step outside "
            f"{expected['first_step']}-{expected['last_step']}"
        )
    return features, labels, feature_names, steps, row_positions


def _build_models(training_labels: np.ndarray) -> dict[str, Any]:
    scale_pos_weight = _scale_pos_weight(training_labels)

    xgboost_module = importlib.import_module("xgboost")
    xgb_classifier = xgboost_module.XGBClassifier
    xgb_configuration = {
        key: value
        for key, value in MODEL_CONFIGURATIONS["XGBoost"].items()
        if key != "class" and key != "scale_pos_weight"
    }
    xgb_configuration["scale_pos_weight"] = scale_pos_weight

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_SEED,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced_subsample",
            max_features="sqrt",
            n_estimators=300,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "XGBoost": xgb_classifier(**xgb_configuration),
    }


def _scale_pos_weight(training_labels: np.ndarray) -> float:
    negative_count = int(np.count_nonzero(training_labels == 0))
    positive_count = int(np.count_nonzero(training_labels == 1))
    if positive_count == 0:
        raise ValueError("Training labels contain no positive fraud examples")
    if negative_count == 0:
        raise ValueError("Training labels contain no legitimate examples")
    return negative_count / positive_count


def _check_disjoint_source_positions(
    training_positions: np.ndarray,
    validation_positions: np.ndarray,
) -> None:
    overlap = np.intersect1d(
        training_positions, validation_positions, assume_unique=True
    )
    if overlap.size:
        sample_positions = overlap[:10].tolist()
        raise ValueError(
            "Training and validation artifacts share "
            f"{overlap.size} source row positions; examples: {sample_positions}. "
            "Refusing to fit any candidate model."
        )


def _positive_class_scores(model: Any, features: np.ndarray) -> np.ndarray:
    class_labels = np.asarray(model.classes_)
    positive_class_indices = np.flatnonzero(class_labels == 1)
    if len(positive_class_indices) != 1:
        raise ValueError(
            f"Expected one positive class in estimator classes, got {class_labels}"
        )
    return model.predict_proba(features)[:, int(positive_class_indices[0])]


def _select_validation_threshold(
    labels: np.ndarray, positive_scores: np.ndarray
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, positive_scores)
    if not len(thresholds):
        raise ValueError("Cannot select a threshold from an empty validation curve")
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    maximum_f1 = float(f1_values.max())
    best_candidates = np.flatnonzero(np.isclose(f1_values, maximum_f1))
    chosen_index = int(best_candidates[-1])
    return float(thresholds[chosen_index]), maximum_f1


def _evaluate_validation(
    labels: np.ndarray,
    positive_scores: np.ndarray,
) -> dict[str, Any]:
    threshold, threshold_f1 = _select_validation_threshold(labels, positive_scores)
    predicted_labels = positive_scores >= threshold
    matrix = confusion_matrix(labels, predicted_labels, labels=[0, 1])
    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(labels, predicted_labels, zero_division=0)),
        "threshold_selection_f1": threshold_f1,
        "average_precision_ap": float(
            average_precision_score(labels, positive_scores)
        ),
        "roc_auc": float(roc_auc_score(labels, positive_scores)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    return metrics


def _render_report(
    result_records: list[dict[str, Any]], selected_model_name: str
) -> str:
    lines = [
        "# Candidate Model Validation Results",
        "",
        "## Scope and prevalence",
        "",
        "All candidate metrics and decision thresholds in this report use the "
        "validation split only. The test artifact was not loaded or evaluated.",
        "",
        "The development sample has **14.4% fraud prevalence**. The test "
        "partition has approximately **13.01%** fraud prevalence, compared "
        "with **1.210811%** in the original population. Validation metrics "
        "describe the enriched development sample, not the original population.",
        "",
        "## Model selection and threshold",
        "",
        f"Selected candidate by highest validation Average Precision (AP): "
        f"**{selected_model_name}**.",
        "Each model threshold maximizes validation F1; if multiple thresholds "
        "tie, the largest tied threshold is selected. No threshold or candidate "
        "selection used test labels or scores.",
        "",
        "Confusion matrices use rows=true labels `[legitimate, fraud]` and "
        "columns=predicted labels `[legitimate, fraud]`.",
        "",
        "## Validation metrics",
        "",
        "| Model | Threshold | Precision | Recall | F1 | Average Precision (AP) | ROC-AUC | Confusion matrix |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in result_records:
        metrics = record["metrics"]
        matrix = metrics["confusion_matrix"]
        lines.append(
            f"| {record['name']} | {metrics['threshold']:.8f} "
            f"| {metrics['precision']:.6f} | {metrics['recall']:.6f} "
            f"| {metrics['f1']:.6f} | "
            f"{metrics['average_precision_ap']:.6f} "
            f"| {metrics['roc_auc']:.6f} | "
            f"`{matrix[0]} / {matrix[1]}` |"
        )
    lines.extend(["", "## Reproducibility and configurations", ""])
    lines.append(f"Random seed: `{RANDOM_SEED}`.")
    for record in result_records:
        lines.extend(
            [
                "",
                f"### {record['name']}",
                "",
                "```json",
                json.dumps(record["configuration"], indent=2),
                "```",
                "",
                f"Model artifact: `{record['artifact_path']}`.",
                f"Associated preprocessing artifact: `{record['preprocessing_path']}`.",
                "Transformed feature names are stored alongside the estimator.",
            ]
        )
    return "\n".join(lines) + "\n"


def _persist_successful_results(
    result_records: list[dict[str, Any]],
    feature_names: list[str],
    preprocessing_relative_path: str,
    report_text: str,
) -> None:
    """Stage every result first, then replace final paths after all candidates pass."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifacts: list[tuple[Path, dict[str, Any]]] = []
    for record in result_records:
        artifacts.append(
            (
                MODELS_DIR / f"{record['safe_name']}.joblib",
                {
                    "estimator": record["estimator"],
                    "selected_validation_threshold": record["metrics"]["threshold"],
                    "selection_metric": (
                        "validation F1; ties choose highest threshold"
                    ),
                    "random_seed": RANDOM_SEED,
                    "configuration": record["configuration"],
                    "feature_names": feature_names,
                    "preprocessing_artifact": preprocessing_relative_path,
                },
            )
        )

    staged_files: list[tuple[Path, Path]] = []
    try:
        for destination, artifact in artifacts:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            staged_files.append((temporary_path, destination))
            joblib.dump(artifact, temporary_path)

        with tempfile.NamedTemporaryFile(
            prefix=f".{REPORT_PATH.name}.",
            suffix=".tmp",
            dir=REPORT_PATH.parent,
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as temporary_report:
            temporary_report.write(report_text)
            report_temporary_path = Path(temporary_report.name)
        staged_files.append((report_temporary_path, REPORT_PATH))

        for temporary_path, destination in staged_files:
            os.replace(temporary_path, destination)
    finally:
        for temporary_path, _ in staged_files:
            if temporary_path.exists():
                temporary_path.unlink()


def main() -> None:
    _check_artifact_paths()
    _require_xgboost()

    train_x, train_y, feature_names, _, training_positions = _load_and_validate_bundle(
        TRAIN_PATH, "train"
    )
    validation_x, validation_y, _, _, validation_positions = _load_and_validate_bundle(
        VALIDATION_PATH, "validation", feature_names
    )
    if validation_x.shape[1] != train_x.shape[1]:
        raise ValueError("Training and validation matrices have incompatible widths")

    _check_disjoint_source_positions(training_positions, validation_positions)

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise TypeError("Preprocessing artifact cannot provide transformed feature names")
    preprocessor_feature_names = preprocessor.get_feature_names_out().tolist()
    if feature_names != preprocessor_feature_names:
        raise ValueError(
            "Prepared feature names do not match the saved preprocessing artifact"
        )

    models = _build_models(train_y)
    result_records: list[dict[str, Any]] = []
    preprocessing_relative_path = str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT))

    for model_name, model in models.items():
        model.fit(train_x, train_y)
        validation_scores = _positive_class_scores(model, validation_x)
        metrics = _evaluate_validation(validation_y, validation_scores)
        safe_name = model_name.lower().replace(" ", "_")
        configuration = dict(MODEL_CONFIGURATIONS[model_name])
        if model_name == "XGBoost":
            configuration["scale_pos_weight"] = _scale_pos_weight(train_y)
        result_records.append(
            {
                "name": model_name,
                "safe_name": safe_name,
                "estimator": model,
                "metrics": metrics,
                "configuration": configuration,
                "artifact_path": str(
                    (MODELS_DIR / f"{safe_name}.joblib").relative_to(PROJECT_ROOT)
                ),
                "preprocessing_path": preprocessing_relative_path,
            }
        )
        print(
            f"{model_name}: threshold={metrics['threshold']:.8f}, "
            f"precision={metrics['precision']:.6f}, "
            f"recall={metrics['recall']:.6f}, F1={metrics['f1']:.6f}, "
            f"AP={metrics['average_precision_ap']:.6f}, "
            f"ROC-AUC={metrics['roc_auc']:.6f}"
        )

    selected = max(
        result_records,
        key=lambda result: result["metrics"]["average_precision_ap"],
    )
    report_text = _render_report(result_records, selected["name"])
    _persist_successful_results(
        result_records,
        feature_names,
        preprocessing_relative_path,
        report_text,
    )
    print(f"Validation report saved: {REPORT_PATH}")
    print(f"Candidate models saved under: {MODELS_DIR}")


if __name__ == "__main__":
    main()
