"""Synthetic unit tests for prediction-service validation and transaction flow."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import src.prediction_service as prediction_service
from src.feature_store import TransactionFeatureStore
from src.preprocessing import MODEL_FEATURE_COLUMNS
from src.prediction_service import (
    EXPECTED_TRANSFORMED_FEATURE_COUNT,
    PredictionService,
)


THRESHOLD = 0.5
FEATURE_NAMES = [f"transformed_{index}" for index in range(39)]
CLASSIFIER_CONFIGURATION = {"class": "xgboost.XGBClassifier"}


class FakeEstimator:
    def __init__(
        self,
        probabilities: np.ndarray | None = None,
        *,
        classes: list[int] | None = None,
        failure_step: int | None = None,
    ) -> None:
        self.classes_ = np.asarray(classes if classes is not None else [0, 1])
        self.n_features_in_ = EXPECTED_TRANSFORMED_FEATURE_COUNT
        self.probabilities = probabilities
        self.failure_step = failure_step
        self.predict_calls = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        if self.failure_step is not None and np.any(features[:, 1] == self.failure_step):
            self.failure_step = None
            raise RuntimeError("synthetic estimator failure")
        if self.probabilities is not None:
            return self.probabilities

        fraud_probabilities = features[:, 0] / 100.0
        if self.classes_.tolist() == [1, 0]:
            return np.column_stack((fraud_probabilities, 1 - fraud_probabilities))
        return np.column_stack((1 - fraud_probabilities, fraud_probabilities))


class FakePreprocessor:
    def __init__(
        self,
        *,
        failure_amount: float | None = None,
        output_width: int = EXPECTED_TRANSFORMED_FEATURE_COUNT,
    ) -> None:
        self.failure_amount = failure_amount
        self.output_width = output_width
        self.transform_calls: list[pd.DataFrame] = []
        self.fit_calls = 0
        self.fit_transform_calls = 0

    def transform(self, model_inputs: pd.DataFrame) -> np.ndarray:
        self.transform_calls.append(model_inputs.copy())
        if (
            self.failure_amount is not None
            and model_inputs["amount"].eq(self.failure_amount).any()
        ):
            raise RuntimeError("synthetic preprocessing failure")

        transformed = np.zeros((len(model_inputs), self.output_width), dtype=float)
        if self.output_width:
            transformed[:, 0] = model_inputs["amount"].to_numpy(dtype=float)
        if self.output_width > 1:
            transformed[:, 1] = model_inputs["step"].to_numpy(dtype=float)
        return transformed

    def fit(self) -> None:
        self.fit_calls += 1
        raise AssertionError("fit must not be called")

    def fit_transform(self) -> None:
        self.fit_transform_calls += 1
        raise AssertionError("fit_transform must not be called")


def _history(rows: list[tuple[int, float, str, str]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        rows or [],
        columns=("step", "amount", "customer", "merchant"),
    ).astype(
        {
            "step": "int64",
            "amount": "float64",
            "customer": "object",
            "merchant": "object",
        }
    )


def _transactions(
    rows: list[tuple[int, float, str, str]],
    *,
    transaction_ids: list[str] | None = None,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    count = len(rows)
    frame = pd.DataFrame(
        {
            "step": [row[0] for row in rows],
            "amount": [row[1] for row in rows],
            "customer": [row[2] for row in rows],
            "merchant": [row[3] for row in rows],
            "category": ["cat-1"] * count,
            "age": ["3"] * count,
            "gender": ["F"] * count,
            "zipcodeOri": ["origin"] * count,
            "zipMerchant": ["merchant-zip"] * count,
        }
    )
    if transaction_ids is not None:
        frame["transaction_id"] = transaction_ids
    if labels is not None:
        frame["fraud"] = labels
    return frame


def _new_service(
    *,
    history: pd.DataFrame | None = None,
    watermark: int = 0,
    estimator: FakeEstimator | None = None,
    preprocessor: FakePreprocessor | None = None,
) -> PredictionService:
    service = PredictionService.__new__(PredictionService)
    service._store = TransactionFeatureStore(
        _history() if history is None else history,
        history_complete_through=watermark,
    )
    service._preprocessor = preprocessor or FakePreprocessor()
    service._estimator = estimator or FakeEstimator()
    service._threshold = THRESHOLD
    service._feature_names = FEATURE_NAMES.copy()
    positive_indices = np.flatnonzero(service._estimator.classes_ == 1)
    service._positive_class_index = int(positive_indices[0]) if len(positive_indices) else 0
    return service


def _valid_artifact(estimator: Any | None = None) -> dict[str, Any]:
    return {
        "estimator": estimator or FakeEstimator(),
        "selected_validation_threshold": THRESHOLD,
        "feature_names": FEATURE_NAMES.copy(),
        "configuration": CLASSIFIER_CONFIGURATION.copy(),
        "preprocessing_artifact": str(
            Path("data") / "processed" / "prepared" / "preprocessing.joblib"
        ),
    }


def _artifact_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact: dict[str, Any],
) -> PredictionService:
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    preprocessor_path = prediction_service.DEFAULT_PREPROCESSOR_PATH.resolve()
    service = PredictionService.__new__(PredictionService)
    service._model_path = model_path
    service._preprocessor_path = preprocessor_path

    def load_model(path: str | Path) -> dict[str, Any]:
        assert Path(path) == model_path
        return artifact

    monkeypatch.setattr(prediction_service.joblib, "load", load_model)
    monkeypatch.setattr(prediction_service, "XGBClassifier", FakeEstimator)
    return service


def test_artifact_loader_rejects_missing_required_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _valid_artifact()
    del artifact["feature_names"]
    service = _artifact_loader(monkeypatch, tmp_path, artifact)

    with pytest.raises(ValueError, match="must be a dictionary"):
        service._load_model_artifact()


@pytest.mark.parametrize(
    ("threshold", "error_type", "message"),
    [
        ("0.5", TypeError, "numeric"),
        (True, TypeError, "numeric"),
        (np.nan, ValueError, "finite and in"),
        (np.inf, ValueError, "finite and in"),
        (-0.01, ValueError, "finite and in"),
        (1.01, ValueError, "finite and in"),
    ],
)
def test_artifact_loader_rejects_invalid_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    threshold: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    artifact = _valid_artifact()
    artifact["selected_validation_threshold"] = threshold
    service = _artifact_loader(monkeypatch, tmp_path, artifact)

    with pytest.raises(error_type, match=message):
        service._load_model_artifact()


@pytest.mark.parametrize(
    ("reference", "error_type", "message"),
    [
        (None, TypeError, "path string"),
        ("outputs/models/xgboost.joblib", ValueError, "different preprocessing"),
    ],
)
def test_artifact_loader_rejects_invalid_preprocessor_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reference: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    artifact = _valid_artifact()
    artifact["preprocessing_artifact"] = reference
    service = _artifact_loader(monkeypatch, tmp_path, artifact)

    with pytest.raises(error_type, match=message):
        service._load_model_artifact()


def test_preprocessor_loader_rejects_missing_referenced_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference = Path("data") / "processed" / "missing-preprocessing.joblib"
    artifact = _valid_artifact()
    artifact["preprocessing_artifact"] = str(reference)
    service = _artifact_loader(monkeypatch, tmp_path, artifact)
    preprocessor_path = (tmp_path / reference).resolve()
    service._preprocessor_path = preprocessor_path
    monkeypatch.setattr(prediction_service, "PROJECT_ROOT", tmp_path)
    load_calls: list[Path] = []

    def load_model_only(path: str | Path) -> dict[str, Any]:
        load_calls.append(Path(path))
        assert Path(path) == service._model_path
        return artifact

    monkeypatch.setattr(prediction_service.joblib, "load", load_model_only)
    service._artifact = service._load_model_artifact()

    with pytest.raises(
        FileNotFoundError,
        match="Fitted preprocessing artifact not found",
    ):
        service._load_preprocessor()

    assert not preprocessor_path.exists()
    assert load_calls == [service._model_path]


@pytest.mark.parametrize(
    ("feature_names", "message"),
    [
        (["feature"] * 39, "non-empty unique strings"),
        ([""] + FEATURE_NAMES[1:], "non-empty unique strings"),
        (["   "] + FEATURE_NAMES[1:], "non-empty unique strings"),
        ([1] + FEATURE_NAMES[1:], "non-empty unique strings"),
        (FEATURE_NAMES[:-1], "exactly 39 transformed feature names"),
    ],
)
def test_artifact_loader_rejects_invalid_feature_names(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    feature_names: list[Any],
    message: str,
) -> None:
    artifact = _valid_artifact()
    artifact["feature_names"] = feature_names
    service = _artifact_loader(monkeypatch, tmp_path, artifact)

    with pytest.raises(ValueError, match=message):
        service._load_model_artifact()


def test_artifact_loader_rejects_a_non_xgboost_estimator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _valid_artifact(estimator=object())
    service = _artifact_loader(monkeypatch, tmp_path, artifact)

    with pytest.raises(TypeError, match="must be an xgboost.XGBClassifier"):
        service._load_model_artifact()


class FakeFittedPreprocessor:
    def __init__(
        self,
        *,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> None:
        self.feature_names_in_ = (
            MODEL_FEATURE_COLUMNS.copy() if input_names is None else input_names
        )
        self.output_names = FEATURE_NAMES.copy() if output_names is None else output_names

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.output_names)


def test_valid_mocked_artifacts_are_accepted_without_loading_saved_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _valid_artifact()
    service = _artifact_loader(monkeypatch, tmp_path, artifact)
    fake_preprocessor = FakeFittedPreprocessor()

    def load_mocked_artifact(path: str | Path) -> Any:
        if Path(path) == service._model_path:
            return artifact
        assert Path(path) == service._preprocessor_path
        return fake_preprocessor

    monkeypatch.setattr(prediction_service.joblib, "load", load_mocked_artifact)
    monkeypatch.setattr(prediction_service, "check_is_fitted", lambda item: item)

    service._artifact = service._load_model_artifact()
    service._preprocessor = service._load_preprocessor()
    service._estimator = service._artifact["estimator"]
    service._feature_names = service._artifact["feature_names"]

    assert service._validate_estimator() == 1
    assert service._preprocessor is fake_preprocessor


@pytest.mark.parametrize(
    ("input_names", "output_names", "message"),
    [
        (["wrong"] * len(MODEL_FEATURE_COLUMNS), FEATURE_NAMES, "input feature names"),
        (
            MODEL_FEATURE_COLUMNS.copy(),
            FEATURE_NAMES[:-1],
            "exactly 39 features",
        ),
        (
            MODEL_FEATURE_COLUMNS.copy(),
            ["wrong"] * 39,
            "do not match the XGBoost artifact",
        ),
    ],
)
def test_preprocessor_loader_rejects_invalid_names_or_width(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    input_names: list[str],
    output_names: list[str],
    message: str,
) -> None:
    service = _artifact_loader(monkeypatch, tmp_path, _valid_artifact())
    service._artifact = _valid_artifact()
    preprocessor_path = tmp_path / "preprocessor.joblib"
    preprocessor_path.touch()
    service._preprocessor_path = preprocessor_path
    fake = FakeFittedPreprocessor(input_names=input_names, output_names=output_names)

    def load_preprocessor(path: str | Path) -> FakeFittedPreprocessor:
        assert Path(path) == preprocessor_path
        return fake

    monkeypatch.setattr(prediction_service.joblib, "load", load_preprocessor)
    monkeypatch.setattr(prediction_service, "check_is_fitted", lambda item: item)

    with pytest.raises(ValueError, match=message):
        service._load_preprocessor()


def test_estimator_validation_requires_callable_predict_proba() -> None:
    service = _new_service()
    service._estimator.predict_proba = None

    with pytest.raises(TypeError, match="callable predict_proba"):
        service._validate_estimator()


@pytest.mark.parametrize("classes", [[0], [0, 2], [1, 2]])
def test_estimator_validation_requires_binary_classes_including_fraud(
    classes: list[int],
) -> None:
    service = _new_service(estimator=FakeEstimator(classes=classes))

    with pytest.raises(ValueError, match="binary classes"):
        service._validate_estimator()


def test_estimator_validation_rejects_reversed_class_order() -> None:
    service = _new_service(estimator=FakeEstimator(classes=[1, 0]))

    with pytest.raises(ValueError, match="binary classes"):
        service._validate_estimator()


def test_estimator_validation_requires_expected_transformed_width() -> None:
    service = _new_service()
    service._estimator.n_features_in_ = 38

    with pytest.raises(ValueError, match="feature count"):
        service._validate_estimator()


def test_estimator_validation_does_not_call_predict_proba() -> None:
    estimator = FakeEstimator()
    service = _new_service(estimator=estimator)

    service._validate_estimator()

    assert estimator.predict_calls == 0


def test_private_helper_selects_class_one_independently_of_artifact_contract() -> None:
    estimator = FakeEstimator(
        classes=[1, 0],
        probabilities=np.asarray([[0.8, 0.2]]),
    )
    service = _new_service(estimator=estimator)

    probabilities = service._positive_class_probabilities(
        np.zeros((1, 39)),
        expected_row_count=1,
    )

    assert probabilities.tolist() == [0.8]


def test_model_inputs_preserve_training_order_and_exclude_identifiers_and_label() -> None:
    service = _new_service()
    transactions = _transactions(
        [(1, 25.0, "customer-secret", "merchant-secret")],
        transaction_ids=["transaction-secret"],
        labels=[1],
    )
    prepared = service._store.prepare_step(transactions)

    model_inputs = service._build_model_inputs(transactions, prepared.features)

    assert list(model_inputs.columns) == MODEL_FEATURE_COLUMNS
    assert not {"customer", "merchant", "transaction_id", "fraud"}.intersection(
        model_inputs.columns
    )
    assert not any(
        "secret" in str(value)
        for value in model_inputs.to_numpy().ravel()
    )


def test_predict_calls_transform_only_and_checks_transformed_width() -> None:
    preprocessor = FakePreprocessor()
    service = _new_service(preprocessor=preprocessor)
    transactions = _transactions([(1, 20.0, "customer-a", "merchant-a")])

    service.predict(transactions)

    assert len(preprocessor.transform_calls) == 1
    assert preprocessor.fit_calls == 0
    assert preprocessor.fit_transform_calls == 0
    assert list(preprocessor.transform_calls[0].columns) == MODEL_FEATURE_COLUMNS


def test_predict_rejects_unexpected_transformed_width_without_committing() -> None:
    service = _new_service(preprocessor=FakePreprocessor(output_width=38))

    with pytest.raises(ValueError, match="invalid shape"):
        service.predict(_transactions([(1, 20.0, "customer-a", "merchant-a")]))

    assert service.history_complete_through == 0


def test_predict_preserves_order_index_ids_threshold_and_classification() -> None:
    service = _new_service()
    transactions = _transactions(
        [
            (2, 80.0, "customer-b", "merchant-b"),
            (1, 20.0, "customer-a", "merchant-a"),
        ],
        transaction_ids=["txn-step-2", "txn-step-1"],
        labels=[0, 1],
    )
    transactions.index = [41, 7]

    results = service.predict(transactions)

    assert results.index.tolist() == [41, 7]
    assert results["transaction_id"].tolist() == ["txn-step-2", "txn-step-1"]
    assert results["step"].tolist() == [2, 1]
    assert results["fraud_probability"].tolist() == [0.8, 0.2]
    assert results["threshold"].tolist() == [THRESHOLD, THRESHOLD]
    assert results["predicted_fraud"].tolist() == [True, False]
    assert service.history_complete_through == 2


def test_same_step_transactions_use_one_pre_step_history_snapshot() -> None:
    service = _new_service(
        history=_history([(0, 10.0, "customer-a", "merchant-a")]),
    )
    transactions = _transactions(
        [
            (1, 20.0, "customer-a", "merchant-a"),
            (1, 30.0, "customer-a", "merchant-a"),
        ]
    )

    service.predict(transactions)

    model_inputs = service._preprocessor.transform_calls[0]
    assert model_inputs["customer_previous_transaction_count"].tolist() == [1.0, 1.0]
    assert model_inputs["customer_previous_amount_total"].tolist() == [10.0, 10.0]
    assert model_inputs["merchant_previous_transaction_count"].tolist() == [1.0, 1.0]


def test_scoring_failure_does_not_commit_any_part_of_current_step() -> None:
    service = _new_service(
        estimator=FakeEstimator(failure_step=1),
    )
    transactions = _transactions(
        [
            (1, 20.0, "customer-a", "merchant-a"),
            (1, 30.0, "customer-b", "merchant-b"),
        ]
    )

    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        service.predict(transactions)

    assert service.history_complete_through == 0
    retry = service.predict(
        _transactions([(1, 10.0, "customer-a", "merchant-a")])
    )
    assert retry.loc[0, "fraud_probability"] == pytest.approx(0.1)
    assert service.history_complete_through == 1


def test_later_step_failure_keeps_earlier_successful_step_committed() -> None:
    service = _new_service(estimator=FakeEstimator(failure_step=2))
    transactions = _transactions(
        [
            (1, 10.0, "customer-a", "merchant-a"),
            (2, 20.0, "customer-a", "merchant-a"),
        ]
    )

    with pytest.raises(RuntimeError, match="synthetic estimator failure"):
        service.predict(transactions)

    assert service.history_complete_through == 1
    next_step = service.predict(
        _transactions([(2, 30.0, "customer-a", "merchant-a")])
    )
    assert next_step.loc[0, "fraud_probability"] == pytest.approx(0.3)
    assert service.history_complete_through == 2


@pytest.mark.parametrize(
    ("transactions", "error_type", "message"),
    [
        (
            _transactions([(2, 1.0, "customer-a", "merchant-a")]),
            ValueError,
            "consecutive steps",
        ),
        (
            _transactions(
                [
                    (1, 1.0, "customer-a", "merchant-a"),
                    (3, 2.0, "customer-b", "merchant-b"),
                ]
            ),
            ValueError,
            "expected step 2",
        ),
        (
            _transactions(
                [
                    (1, 1.0, "customer-a", "merchant-a"),
                    (1, 1.0, "customer-a", "merchant-a"),
                ]
            ),
            ValueError,
            "exact duplicate history records",
        ),
    ],
)
def test_invalid_or_nonconsecutive_batches_are_rejected(
    transactions: pd.DataFrame,
    error_type: type[Exception],
    message: str,
) -> None:
    service = _new_service()

    with pytest.raises(error_type, match=message):
        service.predict(transactions)

    assert service.history_complete_through == 0


def test_empty_transactions_without_required_columns_report_missing_columns() -> None:
    service = _new_service()

    with pytest.raises(ValueError, match="missing required columns"):
        service.predict(pd.DataFrame())

    assert service.history_complete_through == 0


def test_empty_transactions_with_required_columns_are_rejected_as_empty() -> None:
    service = _new_service()

    with pytest.raises(ValueError, match="empty"):
        service.predict(_transactions([]))

    assert service.history_complete_through == 0


def test_missing_transaction_columns_are_rejected() -> None:
    service = _new_service()
    transactions = _transactions([(1, 1.0, "customer-a", "merchant-a")]).drop(
        columns=["category"]
    )

    with pytest.raises(ValueError, match="missing required columns"):
        service.predict(transactions)


def test_feature_generation_failure_does_not_commit() -> None:
    service = _new_service()
    transactions = _transactions([(1, np.nan, "customer-a", "merchant-a")])

    with pytest.raises(TypeError, match="numeric and non-missing"):
        service.predict(transactions)

    assert service.history_complete_through == 0


def test_preprocessing_failure_discards_current_step() -> None:
    service = _new_service(preprocessor=FakePreprocessor(failure_amount=99.0))

    with pytest.raises(RuntimeError, match="synthetic preprocessing failure"):
        service.predict(_transactions([(1, 99.0, "customer-a", "merchant-a")]))

    assert service.history_complete_through == 0


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        (np.asarray([[0.2]]), "unexpected shape"),
        (np.asarray([[np.nan, 1.0]]), "non-finite or out-of-range"),
        (np.asarray([[np.inf, 0.5]]), "non-finite or out-of-range"),
        (np.asarray([[-np.inf, 0.5]]), "non-finite or out-of-range"),
        (np.asarray([[1.2, -0.2]]), "non-finite or out-of-range"),
        (np.asarray([[-0.1, 1.1]]), "non-finite or out-of-range"),
        (
            np.asarray([["invalid", "invalid"]]),
            "non-finite or out-of-range",
        ),
    ],
)
def test_invalid_probability_outputs_discard_current_step(
    probabilities: np.ndarray,
    message: str,
) -> None:
    estimator = FakeEstimator(probabilities=probabilities)
    service = _new_service(estimator=estimator)

    with pytest.raises(ValueError, match=message):
        service.predict(_transactions([(1, 20.0, "customer-a", "merchant-a")]))

    assert service.history_complete_through == 0


@pytest.mark.parametrize(
    "probabilities",
    [
        np.asarray([[0.8, 0.2]]),
        np.asarray([[0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]),
    ],
    ids=["fewer-rows", "more-rows"],
)
def test_probability_row_count_mismatch_discards_prepared_step(
    probabilities: np.ndarray,
) -> None:
    service = _new_service(estimator=FakeEstimator(probabilities=probabilities))
    transactions = _transactions(
        [
            (1, 20.0, "customer-a", "merchant-a"),
            (1, 30.0, "customer-b", "merchant-b"),
        ]
    )

    with pytest.raises(ValueError, match="unexpected shape"):
        service.predict(transactions)

    assert service.history_complete_through == 0
