"""Reusable preprocessing pipeline for fraud-model input features."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURE_COLUMNS = [
    "amount",
    "step",
    "customer_previous_transaction_count",
    "customer_previous_amount_total",
    "customer_previous_amount_mean",
    "customer_steps_since_previous_activity",
    "merchant_previous_transaction_count",
    "merchant_previous_amount_total",
    "merchant_previous_amount_mean",
    "merchant_steps_since_previous_activity",
]
CATEGORICAL_FEATURE_COLUMNS = [
    "category",
    "age",
    "gender",
    "zipcodeOri",
    "zipMerchant",
]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS


def build_preprocessing_pipeline() -> Pipeline:
    """Build a pipeline; callers fit it on training features only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURE_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("features", transformer)])
