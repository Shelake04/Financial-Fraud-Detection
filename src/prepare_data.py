"""Create causal, chronologically split, preprocessed development datasets.

Run this script only after reviewing the feature and split design. It uses the
50,000-row fraud-enriched sample (14.4% fraud), not the original population
(1.210811% fraud). No models are trained or evaluated here.

Outputs are written beneath ``data/processed/prepared/``:
- ``train.joblib``, ``validation.joblib``, ``test.joblib``: transformed feature
  matrices, separate target arrays, original row positions, and steps.
- ``preprocessing.joblib``: pipeline fitted only on the training feature rows.
- ``split_manifest.json``: boundaries, observed counts, and output descriptions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from feature_engineering import generate_behavioral_features
from preprocessing import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_preprocessing_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "banksim_sample.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "prepared"
EXPECTED_COLUMNS = [
    "step",
    "customer",
    "age",
    "gender",
    "zipcodeOri",
    "merchant",
    "zipMerchant",
    "category",
    "amount",
    "fraud",
]
EXPECTED_ROWS = 50_000
EXPECTED_CLASS_COUNTS = {0: 42_800, 1: 7_200}
SPLIT_BOUNDARIES = {
    "train": (0, 130),
    "validation": (131, 155),
    "test": (156, 179),
}
EXPECTED_SPLIT_COUNTS = {
    "train": {"rows": 35_047, "fraud": 5_240, "legitimate": 29_807},
    "validation": {"rows": 7_572, "fraud": 1_000, "legitimate": 6_572},
    "test": {"rows": 7_381, "fraud": 960, "legitimate": 6_421},
}


def _validate_input(data: pd.DataFrame) -> None:
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected input columns. Expected {EXPECTED_COLUMNS}, "
            f"found {data.columns.tolist()}"
        )
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(data):,}")
    if not is_integer_dtype(data["step"].dtype):
        raise TypeError(f"step must be integer, got {data['step'].dtype}")
    if not is_integer_dtype(data["fraud"].dtype):
        raise TypeError(f"fraud must be integer, got {data['fraud'].dtype}")
    if not is_numeric_dtype(data["amount"].dtype):
        raise TypeError(f"amount must be numeric, got {data['amount'].dtype}")
    if data["fraud"].isna().any() or not data["fraud"].isin([0, 1]).all():
        raise ValueError("fraud must contain only non-missing binary labels 0 and 1")
    observed_class_counts = {
        int(label): int(count)
        for label, count in data["fraud"].value_counts().items()
    }
    if observed_class_counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(
            f"Unexpected class counts: {observed_class_counts}; "
            f"expected {EXPECTED_CLASS_COUNTS}"
        )
    if data[["step", "amount", "customer", "merchant"]].isna().any().any():
        raise ValueError(
            "step, amount, customer, and merchant cannot be missing because "
            "they are required for causal behavioral features"
        )
    if set(data["step"].unique()) != set(range(180)):
        raise ValueError("Expected every step value from 0 through 179")


def _split_positions(data: pd.DataFrame) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for split_name, (first_step, last_step) in SPLIT_BOUNDARIES.items():
        mask = data["step"].between(first_step, last_step).to_numpy()
        positions[split_name] = np.flatnonzero(mask)
    return positions


def _validate_split_counts(
    data: pd.DataFrame, split_positions: dict[str, np.ndarray]
) -> dict[str, dict[str, int | float]]:
    observed: dict[str, dict[str, int | float]] = {}
    combined_positions = np.concatenate(list(split_positions.values()))
    if len(combined_positions) != len(data) or not np.array_equal(
        np.sort(combined_positions), np.arange(len(data))
    ):
        raise RuntimeError("Chronological splits do not partition all input rows once")

    for split_name, row_positions in split_positions.items():
        subset = data.iloc[row_positions]
        rows = len(subset)
        fraud = int(subset["fraud"].eq(1).sum())
        legitimate = int(subset["fraud"].eq(0).sum())
        expected = EXPECTED_SPLIT_COUNTS[split_name]
        actual_counts = {"rows": rows, "fraud": fraud, "legitimate": legitimate}
        if actual_counts != expected:
            raise RuntimeError(
                f"{split_name} count mismatch: actual={actual_counts}, "
                f"expected={expected}"
            )
        first_step, last_step = SPLIT_BOUNDARIES[split_name]
        if not subset["step"].between(first_step, last_step).all():
            raise RuntimeError(f"{split_name} contains rows outside its step bounds")
        observed[split_name] = {
            **actual_counts,
            "fraud_prevalence_percent": fraud / rows * 100,
            "first_step": first_step,
            "last_step": last_step,
        }
    return observed


def _persist_split(
    split_name: str,
    transformed_features: Any,
    labels: np.ndarray,
    row_positions: np.ndarray,
    steps: np.ndarray,
    transformed_feature_names: list[str],
) -> Path:
    output_path = OUTPUT_DIR / f"{split_name}.joblib"
    joblib.dump(
        {
            "X": transformed_features,
            "y": labels,
            "source_row_position": row_positions,
            "step": steps,
            "feature_names": transformed_feature_names,
        },
        output_path,
    )
    return output_path


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Development sample not found: {INPUT_CSV}")

    data = pd.read_csv(INPUT_CSV)
    _validate_input(data)
    split_positions = _split_positions(data)
    split_results = _validate_split_counts(data, split_positions)

    # Build histories over the full chronological sample before partitioning.
    # The feature generator ignores fraud and only uses strictly earlier steps.
    behavioral_features = generate_behavioral_features(data)
    model_features = pd.concat(
        [
            data[["amount", "step"]],
            behavioral_features,
            data[CATEGORICAL_FEATURE_COLUMNS],
        ],
        axis=1,
    )
    if model_features.columns.tolist() != MODEL_FEATURE_COLUMNS:
        raise RuntimeError(
            "Generated model feature columns differ from the declared feature schema"
        )
    if {"fraud", "customer", "merchant"} & set(model_features.columns):
        raise RuntimeError("Target or raw entity IDs leaked into model features")
    if not model_features.index.equals(data.index):
        raise RuntimeError("Feature rows are not aligned to the input dataset")

    train_positions = split_positions["train"]
    train_features = model_features.iloc[train_positions]
    preprocessor = build_preprocessing_pipeline()
    train_matrix = preprocessor.fit_transform(train_features)
    transformed_feature_names = preprocessor.get_feature_names_out().tolist()

    transformed: dict[str, Any] = {"train": train_matrix}
    for split_name in ("validation", "test"):
        row_positions = split_positions[split_name]
        transformed[split_name] = preprocessor.transform(
            model_features.iloc[row_positions]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {
        split_name: str(
            _persist_split(
                split_name,
                transformed[split_name],
                data.iloc[split_positions[split_name]]["fraud"].to_numpy(
                    dtype=np.int8
                ),
                split_positions[split_name],
                data.iloc[split_positions[split_name]]["step"].to_numpy(
                    dtype=np.int64
                ),
                transformed_feature_names,
            ).relative_to(PROJECT_ROOT)
        )
        for split_name in SPLIT_BOUNDARIES
    }
    preprocessing_path = OUTPUT_DIR / "preprocessing.joblib"
    joblib.dump(preprocessor, preprocessing_path)

    manifest = {
        "input": str(INPUT_CSV.relative_to(PROJECT_ROOT)),
        "sample_rows": len(data),
        "sample_columns": EXPECTED_COLUMNS,
        "sample_fraud_prevalence_percent": 14.4,
        "original_population_fraud_prevalence_percent": 1.210811,
        "prevalence_note": (
            "The enriched sample prevalence is not representative of the original "
            "population; group rates are sample-based."
        ),
        "split_counts": split_results,
        "feature_columns_before_encoding": MODEL_FEATURE_COLUMNS,
        "transformed_feature_names": transformed_feature_names,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "target_column": "fraud",
        "target_in_model_inputs": False,
        "behavioral_history": (
            "Computed across the full sample in ascending step order; each step "
            "uses state from strictly earlier steps and updates state only after "
            "all rows at the step have been featured. History is not reset at "
            "train/validation/test boundaries."
        ),
        "outputs": {
            **output_files,
            "preprocessing": str(preprocessing_path.relative_to(PROJECT_ROOT)),
            "manifest": str((OUTPUT_DIR / "split_manifest.json").relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Input: {INPUT_CSV} ({len(data):,} rows)")
    print(
        "Sample fraud prevalence: 14.400000%; "
        "original population prevalence: 1.210811%"
    )
    for split_name, details in split_results.items():
        print(
            f"{split_name}: steps {details['first_step']}-{details['last_step']}, "
            f"rows={details['rows']:,}, fraud={details['fraud']:,}, "
            f"legitimate={details['legitimate']:,}, "
            f"fraud_rate={details['fraud_prevalence_percent']:.6f}%"
        )
    print(f"Numeric features: {len(NUMERIC_FEATURE_COLUMNS)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURE_COLUMNS)}")
    print(f"Prepared datasets, pipeline, and manifest saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
