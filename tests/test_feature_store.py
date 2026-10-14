"""Deterministic unit tests for the in-memory causal transaction feature store."""

import numpy as np
import pandas as pd
import pytest

from src.feature_store import (
    BEHAVIORAL_FEATURE_COLUMNS,
    TransactionFeatureStore,
)


EXPECTED_FEATURE_COLUMNS = (
    "customer_previous_transaction_count",
    "customer_previous_amount_total",
    "customer_previous_amount_mean",
    "customer_steps_since_previous_activity",
    "merchant_previous_transaction_count",
    "merchant_previous_amount_total",
    "merchant_previous_amount_mean",
    "merchant_steps_since_previous_activity",
)


def _records(rows: list[tuple[int, float, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=("step", "amount", "customer", "merchant"))


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": pd.Series(dtype="int64"),
            "amount": pd.Series(dtype="float64"),
            "customer": pd.Series(dtype="object"),
            "merchant": pd.Series(dtype="object"),
        }
    )


def test_customer_features_use_only_prior_steps() -> None:
    history = _records(
        [
            (1, 10.0, "customer-a", "merchant-a"),
            (3, 30.0, "customer-a", "merchant-b"),
        ]
    )
    store = TransactionFeatureStore(history, history_complete_through=3)

    prepared = store.prepare_step(
        _records([(4, 8.0, "customer-a", "merchant-new")])
    )

    assert prepared.features.iloc[0][
        [
            "customer_previous_transaction_count",
            "customer_previous_amount_total",
            "customer_previous_amount_mean",
            "customer_steps_since_previous_activity",
        ]
    ].tolist() == [2.0, 40.0, 20.0, 1.0]


def test_merchant_features_use_only_prior_steps() -> None:
    history = _records(
        [
            (1, 12.0, "customer-a", "merchant-a"),
            (3, 24.0, "customer-b", "merchant-a"),
        ]
    )
    store = TransactionFeatureStore(history, history_complete_through=3)

    prepared = store.prepare_step(
        _records([(4, 8.0, "customer-new", "merchant-a")])
    )

    assert prepared.features.iloc[0][
        [
            "merchant_previous_transaction_count",
            "merchant_previous_amount_total",
            "merchant_previous_amount_mean",
            "merchant_steps_since_previous_activity",
        ]
    ].tolist() == [2.0, 36.0, 18.0, 1.0]


def test_unknown_entities_receive_cold_start_values_with_complete_history() -> None:
    history = _records([(1, 5.0, "known-customer", "known-merchant")])
    store = TransactionFeatureStore(history, history_complete_through=2)

    prepared = store.prepare_step(
        _records([(3, 8.0, "new-customer", "new-merchant")])
    )

    assert prepared.features.iloc[0].tolist() == [0.0, 0.0, 0.0, -1.0] * 2


def test_same_step_rows_share_one_pre_step_snapshot() -> None:
    history = _records([(1, 10.0, "customer-a", "merchant-a")])
    store = TransactionFeatureStore(history, history_complete_through=1)
    transactions = _records(
        [
            (2, 20.0, "customer-a", "merchant-a"),
            (2, 40.0, "customer-a", "merchant-a"),
        ]
    )

    prepared = store.prepare_step(transactions)

    expected_snapshot = [1.0, 10.0, 10.0, 1.0, 1.0, 10.0, 10.0, 1.0]
    assert prepared.features.iloc[0].tolist() == expected_snapshot
    assert prepared.features.iloc[1].tolist() == expected_snapshot

    store.commit_step(prepared)
    next_step = store.prepare_step(
        _records([(3, 1.0, "customer-a", "merchant-a")])
    )
    assert next_step.features.iloc[0].tolist() == [
        3.0,
        70.0,
        pytest.approx(70.0 / 3),
        1.0,
        3.0,
        70.0,
        pytest.approx(70.0 / 3),
        1.0,
    ]


@pytest.mark.parametrize(
    ("watermark", "step"),
    [(2, 4), (4, 4)],
)
def test_prepare_step_requires_exact_preceding_watermark(
    watermark: int,
    step: int,
) -> None:
    store = TransactionFeatureStore(_empty_history(), history_complete_through=watermark)

    with pytest.raises(ValueError, match="explicitly complete"):
        store.prepare_step(_records([(step, 1.0, "customer-a", "merchant-a")]))

    assert store.history_complete_through == watermark


@pytest.mark.parametrize(
    ("rows", "error_type", "message"),
    [
        (
            [(1, np.nan, "customer-a", "merchant-a")],
            TypeError,
            "numeric and non-missing",
        ),
        ([(1, np.inf, "customer-a", "merchant-a")], ValueError, "finite"),
        ([(1, "not-an-amount", "customer-a", "merchant-a")], TypeError, "numeric"),
        ([(1, 1.0, None, "merchant-a")], ValueError, "IDs must not be missing"),
        ([(1, 1.0, "customer-a", "")], ValueError, "must not be empty"),
        ([(1.5, 1.0, "customer-a", "merchant-a")], TypeError, "integer dtype"),
    ],
)
def test_invalid_records_are_rejected(
    rows: list[tuple[object, object, object, object]],
    error_type: type[Exception],
    message: str,
) -> None:
    store = TransactionFeatureStore(_empty_history(), history_complete_through=0)
    records = pd.DataFrame(
        rows,
        columns=("step", "amount", "customer", "merchant"),
    )

    with pytest.raises(error_type, match=message):
        store.prepare_step(records)

    assert store.history_complete_through == 0


def test_exact_duplicate_records_are_rejected() -> None:
    store = TransactionFeatureStore(_empty_history(), history_complete_through=1)
    duplicate_batch = _records(
        [
            (2, 5.0, "customer-a", "merchant-a"),
            (2, 5.0, "customer-a", "merchant-a"),
        ]
    )

    with pytest.raises(ValueError, match="exact duplicate history records"):
        store.prepare_step(duplicate_batch)

    assert store.history_complete_through == 1


def test_discard_step_leaves_history_unchanged() -> None:
    store = TransactionFeatureStore(
        _records([(1, 10.0, "customer-a", "merchant-a")]),
        history_complete_through=1,
    )
    prepared = store.prepare_step(
        _records([(2, 20.0, "customer-a", "merchant-a")])
    )

    store.discard_step(prepared)
    next_prepared = store.prepare_step(
        _records([(2, 30.0, "customer-a", "merchant-a")])
    )

    assert next_prepared.features.iloc[0][
        "customer_previous_transaction_count"
    ] == 1
    assert next_prepared.features.iloc[0]["customer_previous_amount_total"] == 10.0


def test_commit_step_applies_history_only_after_commit() -> None:
    store = TransactionFeatureStore(
        _records([(1, 10.0, "customer-a", "merchant-a")]),
        history_complete_through=1,
    )
    prepared = store.prepare_step(
        _records([(2, 20.0, "customer-a", "merchant-a")])
    )

    assert store.history_complete_through == 1
    with pytest.raises(RuntimeError, match="already prepared"):
        store.prepare_step(_records([(2, 30.0, "customer-a", "merchant-b")]))

    store.commit_step(prepared)

    assert store.history_complete_through == 2
    next_prepared = store.prepare_step(
        _records([(3, 5.0, "customer-a", "merchant-a")])
    )
    assert next_prepared.features.iloc[0][
        "customer_previous_transaction_count"
    ] == 2
    assert next_prepared.features.iloc[0]["customer_previous_amount_total"] == 30.0


def test_repeated_and_foreign_state_transitions_are_rejected() -> None:
    store = TransactionFeatureStore(_empty_history(), history_complete_through=0)
    other_store = TransactionFeatureStore(_empty_history(), history_complete_through=0)
    prepared = store.prepare_step(
        _records([(1, 2.0, "customer-a", "merchant-a")])
    )

    with pytest.raises(ValueError, match="different feature store"):
        other_store.commit_step(prepared)

    store.commit_step(prepared)
    with pytest.raises(ValueError, match="not the active pending step"):
        store.commit_step(prepared)
    with pytest.raises(ValueError, match="not the active pending step"):
        store.discard_step(prepared)
    with pytest.raises(ValueError, match="explicitly complete"):
        store.prepare_step(_records([(1, 3.0, "customer-b", "merchant-b")]))

    assert store.history_complete_through == 1


def test_feature_columns_and_rows_preserve_documented_order_and_alignment() -> None:
    store = TransactionFeatureStore(
        _records([(1, 5.0, "customer-a", "merchant-a")]),
        history_complete_through=1,
    )
    transactions = _records(
        [
            (2, 3.0, "customer-new", "merchant-new"),
            (2, 7.0, "customer-a", "merchant-a"),
        ]
    )
    transactions.index = [21, 8]

    prepared = store.prepare_step(transactions)

    assert BEHAVIORAL_FEATURE_COLUMNS == EXPECTED_FEATURE_COLUMNS
    assert tuple(prepared.features.columns) == EXPECTED_FEATURE_COLUMNS
    assert prepared.features.index.tolist() == [21, 8]
    assert prepared.features.loc[21].tolist() == [0.0, 0.0, 0.0, -1.0] * 2
    assert prepared.features.loc[8].tolist() == [1.0, 5.0, 5.0, 1.0] * 2


def test_fraud_label_is_not_required_or_used() -> None:
    store = TransactionFeatureStore(
        _records([(1, 5.0, "customer-a", "merchant-a")]),
        history_complete_through=1,
    )

    prepared = store.prepare_step(
        pd.DataFrame(
            {
                "step": [2],
                "amount": [3.0],
                "customer": ["customer-a"],
                "merchant": ["merchant-a"],
            }
        )
    )

    assert tuple(prepared.features.columns) == EXPECTED_FEATURE_COLUMNS
    assert "fraud" not in prepared.features.columns


def test_failed_preparation_and_discarded_batch_do_not_change_history() -> None:
    store = TransactionFeatureStore(
        _records([(1, 10.0, "customer-a", "merchant-a")]),
        history_complete_through=1,
    )

    with pytest.raises(ValueError, match="share one step"):
        store.prepare_step(
            _records(
                [
                    (2, 20.0, "customer-a", "merchant-a"),
                    (3, 30.0, "customer-a", "merchant-a"),
                ]
            )
        )

    prepared = store.prepare_step(
        _records([(2, 20.0, "customer-a", "merchant-a")])
    )
    store.discard_step(prepared)
    retry = store.prepare_step(
        _records([(2, 40.0, "customer-a", "merchant-a")])
    )

    assert retry.features.iloc[0][
        "customer_previous_transaction_count"
    ] == 1
    assert retry.features.iloc[0]["customer_previous_amount_total"] == 10.0
