"""Synthetic UI tests for the prediction demonstration."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import src.prediction_service as prediction_service
from app import app as dashboard_app


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"
DEMO_FIELDS = {
    "demo_step": 1,
    "demo_amount": 42.75,
    "demo_age": "3",
    "demo_gender": "F",
    "demo_category": "synthetic-category",
    "demo_zipcode_ori": "synthetic-origin",
    "demo_zip_merchant": "synthetic-merchant-zip",
    "demo_customer": "synthetic-customer",
    "demo_merchant": "synthetic-merchant",
}


class MockPredictionService:
    instances: list["MockPredictionService"] = []
    fail_initialization = False
    fail_prediction = False

    def __init__(
        self,
        history: pd.DataFrame,
        *,
        history_complete_through: int,
    ) -> None:
        if self.fail_initialization:
            raise RuntimeError("mock artifact-loading failure")
        self.history = history.copy()
        self.history_complete_through = history_complete_through
        self.threshold = 0.7
        self.transactions: pd.DataFrame | None = None
        self.instances.append(self)

    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if self.fail_prediction:
            raise RuntimeError("mock scoring failure")
        self.transactions = transactions.copy()
        return pd.DataFrame(
            {
                "step": transactions["step"].to_numpy(),
                "transaction_id": transactions["transaction_id"].to_numpy(),
                "fraud_probability": [0.25],
                "threshold": [self.threshold],
                "predicted_fraud": [False],
            },
            index=transactions.index.copy(),
        )


@pytest.fixture(autouse=True)
def mock_prediction_service(monkeypatch: pytest.MonkeyPatch) -> None:
    MockPredictionService.instances = []
    MockPredictionService.fail_initialization = False
    MockPredictionService.fail_prediction = False
    monkeypatch.setattr(
        prediction_service,
        "PredictionService",
        MockPredictionService,
    )


def _demo_page() -> AppTest:
    app = AppTest.from_file(str(APP_PATH)).run()
    app.radio[0].set_value("Fraud Prediction").run()
    return app


def _submit_demo(app: AppTest, fields: dict[str, Any] | None = None) -> AppTest:
    for key, value in (fields or DEMO_FIELDS).items():
        if key == "demo_step" or key == "demo_amount":
            app.number_input(key=key).set_value(value)
        else:
            app.text_input(key=key).set_value(value)
    app.run()
    submit = next(
        button for button in app.button
        if button.label == "Run synthetic demonstration"
    )
    submit.click().run()
    return app


def test_synthetic_demo_maps_form_to_service_and_displays_result() -> None:
    app = _submit_demo(_demo_page())

    assert not app.exception
    assert len(MockPredictionService.instances) == 1
    service = MockPredictionService.instances[0]
    assert service.history.empty
    assert service.history.columns.tolist() == [
        "step",
        "amount",
        "customer",
        "merchant",
    ]
    assert service.history_complete_through == 0
    assert service.transactions is not None
    assert service.transactions.columns.tolist() == [
        "amount",
        "step",
        "category",
        "age",
        "gender",
        "zipcodeOri",
        "zipMerchant",
        "customer",
        "merchant",
        "transaction_id",
    ]
    transaction = service.transactions.iloc[0]
    assert transaction["amount"] == pytest.approx(42.75)
    assert transaction["step"] == 1
    assert transaction["category"] == "synthetic-category"
    assert transaction["age"] == "3"
    assert transaction["gender"] == "F"
    assert transaction["zipcodeOri"] == "synthetic-origin"
    assert transaction["zipMerchant"] == "synthetic-merchant-zip"
    assert transaction["customer"] == "synthetic-customer"
    assert transaction["merchant"] == "synthetic-merchant"
    assert str(transaction["transaction_id"]).startswith("synthetic-")
    assert "fraud" not in service.transactions.columns
    assert not any(
        column.startswith(("customer_previous_", "merchant_previous_"))
        or column.endswith("_steps_since_previous_activity")
        for column in service.transactions.columns
    )
    assert [metric.label for metric in app.metric][-3:] == [
        "Fraud probability",
        "Stored threshold",
        "Threshold-based classification",
    ]
    assert [metric.value for metric in app.metric][-3:] == [
        "0.250000",
        "0.700000",
        "Not flagged",
    ]
    operational_button = next(
        button for button in app.button
        if button.label == "Operational prediction unavailable"
    )
    assert operational_button.disabled
    assert any("Synthetic demonstration only" in item.value for item in app.warning)


def test_each_demo_submission_generates_a_unique_transaction_id() -> None:
    app = _submit_demo(_demo_page())
    first_id = MockPredictionService.instances[-1].transactions.iloc[0][
        "transaction_id"
    ]

    app = _submit_demo(app)
    second_id = MockPredictionService.instances[-1].transactions.iloc[0][
        "transaction_id"
    ]

    assert first_id != second_id
    assert len(MockPredictionService.instances) == 2


def test_invalid_form_input_is_reported_without_constructing_service() -> None:
    invalid_fields = {**DEMO_FIELDS, "demo_customer": "  "}

    app = _submit_demo(_demo_page(), invalid_fields)

    assert not MockPredictionService.instances
    assert any(
        "Input validation error" in error.value
        and "customer must be a non-empty value" in error.value
        for error in app.error
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step", 0, "supports step 1 only"),
        ("amount", -0.01, "finite, non-negative"),
        ("amount", float("nan"), "finite, non-negative"),
        ("amount", float("inf"), "finite, non-negative"),
        ("age", "  ", "age must be a non-empty value"),
        ("gender", "", "gender must be a non-empty value"),
        ("category", " ", "category must be a non-empty value"),
        ("zipcodeOri", "", "zipcodeOri must be a non-empty value"),
        ("zipMerchant", "", "zipMerchant must be a non-empty value"),
        ("customer", "", "customer must be a non-empty value"),
        ("merchant", "", "merchant must be a non-empty value"),
    ],
)
def test_demo_transaction_builder_validates_step_amount_and_text_fields(
    field: str,
    value: Any,
    message: str,
) -> None:
    inputs = {
        "step": 1,
        "amount": 10.0,
        "age": "3",
        "gender": "F",
        "category": "synthetic-category",
        "zipcodeOri": "synthetic-origin",
        "zipMerchant": "synthetic-merchant-zip",
        "customer": "synthetic-customer",
        "merchant": "synthetic-merchant",
    }
    inputs[field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        dashboard_app._build_synthetic_demo_transaction(inputs)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("initialization", "could not load or validate the model artifacts"),
        ("prediction", "scoring error"),
    ],
)
def test_service_errors_are_reported_without_classification(
    failure: str,
    message: str,
) -> None:
    if failure == "initialization":
        MockPredictionService.fail_initialization = True
    else:
        MockPredictionService.fail_prediction = True

    app = _submit_demo(_demo_page())

    assert any(message in error.value for error in app.error)
    assert not any(
        metric.label == "Threshold-based classification"
        for metric in app.metric
    )
