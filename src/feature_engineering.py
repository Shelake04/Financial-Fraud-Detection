"""Generate causal customer and merchant history features.

Features for a transaction use state from strictly earlier steps. Transactions
sharing a step all receive features from the same pre-step state; state updates
are applied only after every transaction at that step has been featured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


REQUIRED_COLUMNS = {"step", "amount", "customer", "merchant"}
ENTITY_NAMES = ("customer", "merchant")
HISTORY_MEANING = {
    "first_observed_step": (
        "No earlier history: previous count, amount total, and amount mean are "
        "0; steps since previous activity is -1."
    ),
    "same_step": (
        "All rows at a step use state from before that step; transactions within "
        "the step do not contribute to one another's features."
    ),
}


def generate_behavioral_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return past-only customer/merchant features aligned to ``data`` rows.

    The function intentionally selects only step, amount, and entity keys.
    Target labels are neither required nor read. Feature history is continuous
    across all steps in ``data``; callers should pass the full chronologically
    sorted development sample before partitioning the resulting rows.
    """
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing behavioral feature inputs: {sorted(missing_columns)}"
        )
    if data.empty:
        raise ValueError("Cannot generate behavioral features for an empty dataset")
    if not is_integer_dtype(data["step"].dtype):
        raise TypeError(f"step must have an integer dtype, got {data['step'].dtype}")
    if not is_numeric_dtype(data["amount"].dtype):
        raise TypeError(
            f"amount must have a numeric dtype, got {data['amount'].dtype}"
        )
    if data[["step", "amount", "customer", "merchant"]].isna().any().any():
        raise ValueError(
            "step, amount, customer, and merchant must be non-missing to build "
            "causal history"
        )
    amounts = data["amount"].to_numpy(dtype=float)
    if not np.isfinite(amounts).all():
        raise ValueError("amount must contain only finite values")

    source = data[["step", "amount", "customer", "merchant"]].reset_index(drop=True)
    feature_values = {
        f"{entity}_{suffix}": np.zeros(len(source), dtype=np.float64)
        for entity in ENTITY_NAMES
        for suffix in (
            "previous_transaction_count",
            "previous_amount_total",
            "previous_amount_mean",
            "steps_since_previous_activity",
        )
    }
    for entity in ENTITY_NAMES:
        feature_values[f"{entity}_steps_since_previous_activity"].fill(-1)

    # Entity state holds only history already observed: (count, amount sum, step).
    states: dict[str, dict[object, tuple[int, float, int]]] = {
        entity: {} for entity in ENTITY_NAMES
    }

    for step_value, positions in source.groupby("step", sort=True).indices.items():
        step = int(step_value)
        step_positions = np.asarray(positions, dtype=np.int64)

        for entity in ENTITY_NAMES:
            entity_values = source[entity].to_numpy()
            state = states[entity]
            for position in step_positions:
                key = entity_values[position]
                previous_count, previous_total, previous_step = state.get(
                    key, (0, 0.0, -1)
                )
                feature_values[f"{entity}_previous_transaction_count"][position] = (
                    previous_count
                )
                feature_values[f"{entity}_previous_amount_total"][position] = (
                    previous_total
                )
                feature_values[f"{entity}_previous_amount_mean"][position] = (
                    previous_total / previous_count if previous_count else 0.0
                )
                feature_values[
                    f"{entity}_steps_since_previous_activity"
                ][position] = step - previous_step if previous_count else -1

        # Aggregate the whole step first; no row in this step sees these updates.
        for entity in ENTITY_NAMES:
            step_totals: dict[object, list[float]] = {}
            entity_values = source[entity].to_numpy()
            for position in step_positions:
                key = entity_values[position]
                aggregate = step_totals.setdefault(key, [0.0, 0.0])
                aggregate[0] += 1
                aggregate[1] += amounts[position]

            state = states[entity]
            for key, (step_count, step_total) in step_totals.items():
                previous_count, previous_total, _ = state.get(key, (0, 0.0, -1))
                state[key] = (
                    previous_count + int(step_count),
                    previous_total + step_total,
                    step,
                )

    return pd.DataFrame(feature_values, index=data.index)
