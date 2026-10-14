"""In-memory causal history store for customer and merchant features.

This module computes behavioral features only. It does not load models,
preprocess model inputs, infer fraud, or persist state between processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "amount",
    "customer",
    "merchant",
)
OPTIONAL_TRANSACTION_ID: Final[str] = "transaction_id"
ENTITY_NAMES: Final[tuple[str, ...]] = ("customer", "merchant")
FEATURE_SUFFIXES: Final[tuple[str, ...]] = (
    "previous_transaction_count",
    "previous_amount_total",
    "previous_amount_mean",
    "steps_since_previous_activity",
)
BEHAVIORAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{entity}_{suffix}"
    for entity in ENTITY_NAMES
    for suffix in FEATURE_SUFFIXES
)


@dataclass(frozen=True)
class PreparedStep:
    """Feature rows and deferred history update for one complete step."""

    step: int
    features: pd.DataFrame
    _transactions: pd.DataFrame
    _store_id: UUID


class TransactionFeatureStore:
    """Maintain causal customer and merchant state in memory.

    ``history_complete_through`` is an explicit watermark: the caller asserts
    the supplied history is complete for every step through that integer,
    including steps with no records. To prepare step ``s``, the watermark must
    be exactly ``s - 1``. A new entity then receives the established cold-start
    values only under this completeness assertion.

    The caller must call :meth:`commit_step` only after all transactions in a
    prepared step have been successfully scored. On scoring failure, call
    :meth:`discard_step`; neither operation partially updates the state.

    This store is process-local and is not thread-safe. It provides neither
    durable persistence nor concurrent update coordination.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        *,
        history_complete_through: int,
    ) -> None:
        self._validate_watermark(history_complete_through)
        self._history_complete_through = history_complete_through
        self._customer_state: dict[str, tuple[int, float, int]] = {}
        self._merchant_state: dict[str, tuple[int, float, int]] = {}
        self._seen_transaction_ids: set[str] = set()
        self._store_id = uuid4()
        self._pending_step: PreparedStep | None = None

        validated_history = self._validate_records(history, "history")
        if not validated_history.empty:
            if (validated_history["step"] > history_complete_through).any():
                raise ValueError(
                    "History contains records after history_complete_through"
                )
            self._seed_state(validated_history)
            if OPTIONAL_TRANSACTION_ID in validated_history.columns:
                self._seen_transaction_ids.update(
                    validated_history[OPTIONAL_TRANSACTION_ID]
                )

    @property
    def history_complete_through(self) -> int:
        """Return the latest step declared complete in the current state."""
        return self._history_complete_through

    def prepare_step(self, transactions: pd.DataFrame) -> PreparedStep:
        """Build behavioral features for every row at the next complete step.

        No history is updated by this method. All rows must share a single
        step, and every feature uses only state accumulated before that step.
        """
        if self._pending_step is not None:
            raise RuntimeError(
                "A step is already prepared; commit it after successful scoring "
                "or discard it before preparing another step"
            )

        records = self._validate_records(transactions, "transactions")
        if records.empty:
            raise ValueError("Cannot prepare features for an empty transaction step")
        if OPTIONAL_TRANSACTION_ID in records.columns:
            repeated_ids = self._seen_transaction_ids.intersection(
                records[OPTIONAL_TRANSACTION_ID]
            )
            if repeated_ids:
                examples = sorted(repeated_ids)[:5]
                raise ValueError(
                    "Transactions repeat IDs already present in history: "
                    f"{examples}"
                )
        distinct_steps = records["step"].unique()
        if len(distinct_steps) != 1:
            raise ValueError(
                "All transactions passed to prepare_step must share one step"
            )

        step = int(distinct_steps[0])
        required_watermark = step - 1
        if self._history_complete_through != required_watermark:
            raise ValueError(
                f"History must be explicitly complete through step "
                f"{required_watermark} before preparing step {step}; current "
                f"watermark is {self._history_complete_through}"
            )

        features = self._build_features(records, step)
        prepared = PreparedStep(
            step=step,
            features=features,
            _transactions=records,
            _store_id=self._store_id,
        )
        self._pending_step = prepared
        return prepared

    def commit_step(self, prepared: PreparedStep) -> None:
        """Apply the whole prepared step after the caller has scored all rows."""
        self._validate_pending(prepared)

        customer_state = self._customer_state.copy()
        merchant_state = self._merchant_state.copy()
        self._apply_updates(
            customer_state,
            prepared._transactions,
            "customer",
            prepared.step,
        )
        self._apply_updates(
            merchant_state,
            prepared._transactions,
            "merchant",
            prepared.step,
        )

        self._customer_state = customer_state
        self._merchant_state = merchant_state
        if OPTIONAL_TRANSACTION_ID in prepared._transactions.columns:
            self._seen_transaction_ids.update(
                prepared._transactions[OPTIONAL_TRANSACTION_ID]
            )
        self._history_complete_through = prepared.step
        self._pending_step = None

    def discard_step(self, prepared: PreparedStep) -> None:
        """Discard a prepared step without changing history state."""
        self._validate_pending(prepared)
        self._pending_step = None

    def _validate_pending(self, prepared: PreparedStep) -> None:
        if prepared._store_id != self._store_id:
            raise ValueError("Prepared step belongs to a different feature store")
        if self._pending_step is not prepared:
            raise ValueError("Prepared step is not the active pending step")
        if prepared.step != self._history_complete_through + 1:
            raise RuntimeError("Prepared step no longer follows the history watermark")

    def _seed_state(self, history: pd.DataFrame) -> None:
        for entity in ENTITY_NAMES:
            grouped = history.groupby(entity, sort=False).agg(
                transaction_count=("amount", "size"),
                amount_total=("amount", "sum"),
                last_step=("step", "max"),
            )
            if not np.isfinite(grouped["amount_total"].to_numpy(dtype=np.float64)).all():
                raise ValueError(
                    f"History amount totals overflowed while aggregating {entity} IDs"
                )
            state = (
                self._customer_state if entity == "customer" else self._merchant_state
            )
            for entity_id, row in grouped.iterrows():
                state[str(entity_id)] = (
                    int(row["transaction_count"]),
                    float(row["amount_total"]),
                    int(row["last_step"]),
                )

    def _validate_records(
        self,
        records: pd.DataFrame,
        records_name: str,
    ) -> pd.DataFrame:
        if not isinstance(records, pd.DataFrame):
            raise TypeError(f"{records_name} must be a pandas DataFrame")
        missing_columns = set(HISTORY_COLUMNS) - set(records.columns)
        if missing_columns:
            raise ValueError(
                f"{records_name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )
        if records.empty:
            return records.loc[:, list(HISTORY_COLUMNS)].copy()
        if records[["customer", "merchant"]].isna().any().any():
            raise ValueError(
                f"{records_name} customer and merchant IDs must not be missing"
            )
        for entity in ENTITY_NAMES:
            if not records[entity].map(lambda value: isinstance(value, str)).all():
                raise TypeError(
                    f"{records_name} {entity} IDs must be strings; "
                    "normalize identifiers before loading history"
                )
            if records[entity].str.strip().eq("").any():
                raise ValueError(f"{records_name} {entity} IDs must not be empty")
        if records["step"].isna().any() or not is_integer_dtype(records["step"].dtype):
            raise TypeError(f"{records_name} step values must have integer dtype")
        if records["amount"].isna().any() or not is_numeric_dtype(
            records["amount"].dtype
        ):
            raise TypeError(f"{records_name} amounts must be numeric and non-missing")
        amounts = records["amount"].to_numpy(dtype=np.float64)
        if not np.isfinite(amounts).all():
            raise ValueError(f"{records_name} amounts must all be finite")

        selected_columns = list(HISTORY_COLUMNS)
        if OPTIONAL_TRANSACTION_ID in records.columns:
            transaction_ids = records[OPTIONAL_TRANSACTION_ID]
            if transaction_ids.isna().any():
                raise ValueError(
                    f"{records_name} transaction IDs must not be missing"
                )
            if not transaction_ids.map(lambda value: isinstance(value, str)).all():
                raise TypeError(
                    f"{records_name} transaction IDs must be strings"
                )
            if transaction_ids.str.strip().eq("").any():
                raise ValueError(
                    f"{records_name} transaction IDs must not be empty"
                )
            if transaction_ids.duplicated().any():
                raise ValueError(
                    f"{records_name} contains duplicate transaction IDs"
                )
            selected_columns.append(OPTIONAL_TRANSACTION_ID)
        elif records.duplicated(subset=list(HISTORY_COLUMNS)).any():
            raise ValueError(
                f"{records_name} contains exact duplicate history records. "
                f"Supply unique {OPTIONAL_TRANSACTION_ID} values to distinguish "
                "legitimate identical transactions from duplicate ingestion."
            )

        return records.loc[:, selected_columns].copy()

    def _build_features(self, records: pd.DataFrame, step: int) -> pd.DataFrame:
        feature_values = {
            column: np.zeros(len(records), dtype=np.float64)
            for column in BEHAVIORAL_FEATURE_COLUMNS
        }
        for entity in ENTITY_NAMES:
            last_step_column = f"{entity}_steps_since_previous_activity"
            feature_values[last_step_column].fill(-1)
            state = (
                self._customer_state if entity == "customer" else self._merchant_state
            )
            for row_position, entity_id in enumerate(records[entity]):
                previous_count, previous_total, previous_step = state.get(
                    entity_id, (0, 0.0, -1)
                )
                feature_values[
                    f"{entity}_previous_transaction_count"
                ][row_position] = previous_count
                feature_values[f"{entity}_previous_amount_total"][row_position] = (
                    previous_total
                )
                feature_values[f"{entity}_previous_amount_mean"][row_position] = (
                    previous_total / previous_count if previous_count else 0.0
                )
                feature_values[last_step_column][row_position] = (
                    step - previous_step if previous_count else -1
                )

        features = pd.DataFrame(feature_values, index=records.index)
        if features.shape[0] != len(records):
            raise RuntimeError("Behavioral features lost transaction rows")
        return features

    @staticmethod
    def _apply_updates(
        state: dict[str, tuple[int, float, int]],
        transactions: pd.DataFrame,
        entity: str,
        step: int,
    ) -> None:
        grouped = transactions.groupby(entity, sort=False).agg(
            transaction_count=("amount", "size"),
            amount_total=("amount", "sum"),
        )
        if not np.isfinite(grouped["amount_total"].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Step amount totals overflowed while updating {entity}")
        for entity_id, row in grouped.iterrows():
            previous_count, previous_total, _ = state.get(
                str(entity_id), (0, 0.0, -1)
            )
            updated_total = previous_total + float(row["amount_total"])
            if not np.isfinite(updated_total):
                raise ValueError(f"Accumulated amount total overflowed for {entity}")
            state[str(entity_id)] = (
                previous_count + int(row["transaction_count"]),
                updated_total,
                step,
            )

    @staticmethod
    def _validate_watermark(watermark: int) -> None:
        if isinstance(watermark, bool) or not isinstance(watermark, (int, np.integer)):
            raise TypeError("history_complete_through must be an integer step")
