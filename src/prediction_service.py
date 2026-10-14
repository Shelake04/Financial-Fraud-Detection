"""Causal transaction scoring with the saved XGBoost model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from src.feature_store import TransactionFeatureStore
from src.preprocessing import MODEL_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / "xgboost.joblib"
DEFAULT_PREPROCESSOR_PATH = (
    PROJECT_ROOT / "data" / "processed" / "prepared" / "preprocessing.joblib"
)
EXPECTED_TRANSFORMED_FEATURE_COUNT = 39
BEHAVIORAL_COLUMNS = (
    "customer_previous_transaction_count",
    "customer_previous_amount_total",
    "customer_previous_amount_mean",
    "customer_steps_since_previous_activity",
    "merchant_previous_transaction_count",
    "merchant_previous_amount_total",
    "merchant_previous_amount_mean",
    "merchant_steps_since_previous_activity",
)
ENTITY_COLUMNS = ("customer", "merchant")
OPTIONAL_TRANSACTION_ID = "transaction_id"

_REQUIRED_TRANSACTION_COLUMNS = tuple(
    column
    for column in MODEL_FEATURE_COLUMNS
    if column not in BEHAVIORAL_COLUMNS
) + ENTITY_COLUMNS


class PredictionService:
    """Score batches using a fitted preprocessor, saved estimator, and history.

    ``history`` must be trusted, complete operational history, with an explicit
    watermark. The fraud-enriched development sample is not suitable as
    operational history. Completeness is asserted by the caller and cannot be
    independently verified by this service.

    Each call processes complete steps sequentially. All transactions at a step
    share the same pre-step state, and a step is committed only after feature
    transformation and scoring succeed for every transaction in it. If a later
    step fails, previously successful steps from that ``predict()`` call remain
    committed; the failing step is not partially committed.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        *,
        history_complete_through: int,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        preprocessor_path: str | Path = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        self._model_path = Path(model_path).resolve()
        self._preprocessor_path = Path(preprocessor_path).resolve()
        self._artifact = self._load_model_artifact()
        self._preprocessor = self._load_preprocessor()
        self._estimator: XGBClassifier = self._artifact["estimator"]
        self._threshold = float(self._artifact["selected_validation_threshold"])
        self._feature_names: list[str] = self._artifact["feature_names"]
        self._positive_class_index = self._validate_estimator()
        self._store = TransactionFeatureStore(
            history,
            history_complete_through=history_complete_through,
        )

    @property
    def threshold(self) -> float:
        """Return the validation-selected threshold stored in the model artifact."""
        return self._threshold

    @property
    def history_complete_through(self) -> int:
        """Return the feature store's current committed-history watermark."""
        return self._store.history_complete_through

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Score rows from one or more consecutive steps in input order.

        The returned DataFrame preserves the input index and includes ``step``,
        ``fraud_probability``, ``threshold``, and ``predicted_fraud``. If the
        input contains ``transaction_id``, that identifier is returned for
        result matching. Entity IDs and any supplied fraud label are excluded
        from model input.
        """
        records = self._validate_transactions(transactions)
        indexed_records = records.reset_index(drop=True)
        ordered_steps = self._validated_consecutive_steps(indexed_records)

        probabilities = np.empty(len(indexed_records), dtype=np.float64)
        predicted_fraud = np.empty(len(indexed_records), dtype=bool)

        for step in ordered_steps:
            positions = np.flatnonzero(indexed_records["step"].to_numpy() == step)
            step_records = indexed_records.iloc[positions].copy()
            prepared = None
            try:
                prepared = self._store.prepare_step(step_records)
                model_inputs = self._build_model_inputs(step_records, prepared.features)
                transformed = self._preprocessor.transform(model_inputs)
                self._validate_transformed(transformed, len(step_records))
                step_probabilities = self._positive_class_probabilities(
                    transformed,
                    expected_row_count=len(step_records),
                )
                step_predictions = step_probabilities >= self._threshold
                if step_predictions.shape != (len(step_records),):
                    raise ValueError(
                        "Prediction results are not row-aligned with the "
                        "prepared transaction step"
                    )
            except Exception:
                if prepared is not None:
                    self._store.discard_step(prepared)
                raise

            self._store.commit_step(prepared)
            probabilities[positions] = step_probabilities
            predicted_fraud[positions] = step_predictions

        result_data: dict[str, Any] = {
            "step": indexed_records["step"].to_numpy(copy=True),
            "fraud_probability": probabilities,
            "threshold": np.full(len(records), self._threshold, dtype=np.float64),
            "predicted_fraud": predicted_fraud,
        }
        if OPTIONAL_TRANSACTION_ID in indexed_records.columns:
            result_data[OPTIONAL_TRANSACTION_ID] = indexed_records[
                OPTIONAL_TRANSACTION_ID
            ].to_numpy(copy=True)
        return pd.DataFrame(result_data, index=transactions.index.copy())

    def _load_model_artifact(self) -> dict[str, Any]:
        if not self._model_path.is_file():
            raise FileNotFoundError(f"XGBoost model artifact not found: {self._model_path}")
        artifact = joblib.load(self._model_path)
        required_keys = {
            "estimator",
            "selected_validation_threshold",
            "feature_names",
            "configuration",
            "preprocessing_artifact",
        }
        if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
            raise ValueError(
                "XGBoost artifact must be a dictionary containing "
                f"{sorted(required_keys)}"
            )

        estimator = artifact["estimator"]
        if not isinstance(estimator, XGBClassifier):
            raise TypeError(
                "Saved estimator must be an xgboost.XGBClassifier, got "
                f"{type(estimator).__module__}.{type(estimator).__name__}"
            )

        threshold = artifact["selected_validation_threshold"]
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(
            threshold, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Saved validation threshold must be numeric")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(
                f"Saved validation threshold must be finite and in [0, 1], got {threshold}"
            )

        feature_names = artifact["feature_names"]
        if (
            not isinstance(feature_names, list)
            or len(feature_names) != EXPECTED_TRANSFORMED_FEATURE_COUNT
        ):
            raise ValueError(
                "XGBoost artifact must contain exactly "
                f"{EXPECTED_TRANSFORMED_FEATURE_COUNT} transformed feature names"
            )
        if (
            not all(isinstance(name, str) and name.strip() for name in feature_names)
            or len(set(feature_names)) != len(feature_names)
        ):
            raise ValueError(
                "XGBoost artifact feature names must be non-empty unique strings"
            )
        configuration = artifact["configuration"]
        if (
            not isinstance(configuration, dict)
            or configuration.get("class") != "xgboost.XGBClassifier"
        ):
            raise ValueError(
                "XGBoost artifact configuration must identify "
                "xgboost.XGBClassifier"
            )

        preprocessing_reference = artifact["preprocessing_artifact"]
        if not isinstance(preprocessing_reference, str):
            raise TypeError("Preprocessing artifact reference must be a path string")
        referenced_path = (PROJECT_ROOT / preprocessing_reference).resolve()
        if referenced_path != self._preprocessor_path:
            raise ValueError(
                "Model artifact references a different preprocessing artifact: "
                f"{preprocessing_reference}"
            )
        return artifact

    def _load_preprocessor(self) -> Any:
        if not self._preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Fitted preprocessing artifact not found: {self._preprocessor_path}"
            )
        preprocessor = joblib.load(self._preprocessor_path)
        check_is_fitted(preprocessor)
        if not hasattr(preprocessor, "get_feature_names_out"):
            raise TypeError(
                "Preprocessing artifact cannot provide transformed feature names"
            )
        if not hasattr(preprocessor, "feature_names_in_"):
            raise ValueError(
                "Preprocessing artifact does not record its input feature names"
            )
        preprocessor_input_names = list(preprocessor.feature_names_in_)
        if preprocessor_input_names != MODEL_FEATURE_COLUMNS:
            raise ValueError(
                "Preprocessing input feature names or ordering do not match "
                "MODEL_FEATURE_COLUMNS"
            )

        transformed_names = list(preprocessor.get_feature_names_out())
        if len(transformed_names) != EXPECTED_TRANSFORMED_FEATURE_COUNT:
            raise ValueError(
                "Preprocessing artifact must produce exactly "
                f"{EXPECTED_TRANSFORMED_FEATURE_COUNT} features; got "
                f"{len(transformed_names)}"
            )
        if transformed_names != self._artifact["feature_names"]:
            raise ValueError(
                "Preprocessing transformed feature names or ordering do not "
                "match the XGBoost artifact"
            )
        return preprocessor

    def _validate_estimator(self) -> int:
        predict_proba = getattr(self._estimator, "predict_proba", None)
        if not callable(predict_proba):
            raise TypeError(
                "Saved XGBoost estimator must provide a callable predict_proba(); "
                "use a compatible fitted XGBClassifier artifact"
            )
        classes = np.asarray(self._estimator.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1 or not np.array_equal(classes, [0, 1]):
            raise ValueError(
                f"XGBoost estimator must expose binary classes [0, 1], got {classes}"
            )
        if getattr(self._estimator, "n_features_in_", None) != len(self._feature_names):
            raise ValueError(
                "XGBoost estimator input feature count does not match the "
                "stored transformed feature names"
            )
        return int(positive_indices[0])

    @staticmethod
    def _validate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        missing_columns = set(_REQUIRED_TRANSACTION_COLUMNS) - set(transactions.columns)
        if missing_columns:
            raise ValueError(
                "transactions are missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if transactions.empty:
            raise ValueError("Cannot predict an empty transaction batch")
        return transactions.copy()

    def _validated_consecutive_steps(self, records: pd.DataFrame) -> list[int]:
        steps = records["step"]
        if steps.isna().any() or not pd.api.types.is_integer_dtype(steps.dtype):
            raise TypeError("transactions step values must have integer dtype")
        ordered_steps = sorted(int(value) for value in steps.unique())
        expected_step = self._store.history_complete_through + 1
        for step in ordered_steps:
            if step != expected_step:
                raise ValueError(
                    "Prediction batches must cover consecutive steps beginning "
                    f"at {self._store.history_complete_through + 1}; expected "
                    f"step {expected_step}, got {step}"
                )
            expected_step += 1
        return ordered_steps

    @staticmethod
    def _build_model_inputs(
        transactions: pd.DataFrame,
        behavioral_features: pd.DataFrame,
    ) -> pd.DataFrame:
        if len(transactions) != len(behavioral_features):
            raise ValueError(
                "Behavioral features are not row-aligned with the transaction batch"
            )
        if not transactions.index.equals(behavioral_features.index):
            raise ValueError(
                "Behavioral feature index does not match the transaction batch"
            )
        nonbehavioral_columns = [
            column
            for column in MODEL_FEATURE_COLUMNS
            if column not in BEHAVIORAL_COLUMNS
        ]
        model_inputs = transactions.loc[:, nonbehavioral_columns].copy()
        for column in BEHAVIORAL_COLUMNS:
            model_inputs[column] = behavioral_features[column]
        model_inputs = model_inputs.loc[:, MODEL_FEATURE_COLUMNS]
        if list(model_inputs.columns) != MODEL_FEATURE_COLUMNS:
            raise RuntimeError("Model input feature order differs from training")
        return model_inputs

    def _validate_transformed(self, transformed: Any, row_count: int) -> None:
        if not hasattr(transformed, "shape") or transformed.shape != (
            row_count,
            EXPECTED_TRANSFORMED_FEATURE_COUNT,
        ):
            shape = getattr(transformed, "shape", None)
            raise ValueError(
                "Preprocessor output has invalid shape "
                f"{shape}; expected ({row_count}, "
                f"{EXPECTED_TRANSFORMED_FEATURE_COUNT})"
            )
        values = transformed.data if sparse.issparse(transformed) else np.asarray(transformed)
        if not np.isfinite(values).all():
            raise ValueError("Preprocessor output contains non-finite values")

    def _positive_class_probabilities(
        self,
        transformed: Any,
        *,
        expected_row_count: int,
    ) -> np.ndarray:
        probabilities = np.asarray(self._estimator.predict_proba(transformed))
        expected_shape = (expected_row_count, len(self._estimator.classes_))
        if probabilities.shape != expected_shape:
            raise ValueError(
                "XGBoost predict_proba returned an unexpected shape "
                f"{probabilities.shape}; expected {expected_shape} to match "
                "the prepared transaction step"
            )
        if (
            not np.issubdtype(probabilities.dtype, np.number)
            or np.issubdtype(probabilities.dtype, np.complexfloating)
            or not np.isfinite(probabilities).all()
            or not np.all((probabilities >= 0) & (probabilities <= 1))
        ):
            raise ValueError(
                "XGBoost returned non-finite or out-of-range class probabilities; "
                "values must be real numeric probabilities in [0, 1]"
            )
        fraud_probabilities = probabilities[:, self._positive_class_index]
        return fraud_probabilities
