"""Create a reproducible, fraud-enriched BankSim sample for development.

All fraud rows are retained. Legitimate rows are sampled by step, reserving one
row for every step with legitimate transactions and allocating the remaining
quota proportionally to each step's remaining legitimate-row capacity. The
fixed seed makes row selection reproducible. The resulting fraud prevalence is
intentionally higher than the population prevalence, so this sample must not
be used as the final unbiased test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CSV = Path(r"D:\Fraud Detection\bs140513_032310.csv")
OUTPUT_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "banksim_sample.csv"
RANDOM_SEED = 42
TARGET_ROW_COUNT = 50_000
EXPECTED_FRAUD_COUNT = 7_200
EXPECTED_SOURCE_ROW_COUNT = 594_643
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


def _allocate_legitimate_quotas(counts: pd.Series, target_count: int) -> pd.Series:
    """Allocate integer step quotas proportionally, guaranteeing coverage."""
    if target_count < len(counts):
        raise ValueError("Target legitimate-row count is too small to cover every step")
    if target_count > int(counts.sum()):
        raise ValueError("Target legitimate-row count exceeds the source population")

    remaining = target_count - len(counts)
    capacity = counts - 1
    capacity_total = int(capacity.sum())
    if remaining and not capacity_total:
        raise ValueError("Cannot allocate additional legitimate rows across steps")

    quotas = pd.Series(1, index=counts.index, dtype="int64")
    if not remaining:
        return quotas

    exact_extras = capacity * (remaining / capacity_total)
    extra_quotas = np.floor(exact_extras).astype("int64")
    quotas += extra_quotas
    remainder = remaining - int(extra_quotas.sum())
    if remainder:
        fractional_parts = exact_extras - extra_quotas
        priority = fractional_parts.sort_values(ascending=False, kind="stable").index
        quotas.loc[priority[:remainder]] += 1

    if (quotas > counts).any() or int(quotas.sum()) != target_count:
        raise RuntimeError("Step quota allocation did not meet its count constraints")
    return quotas


def _select_row_positions(frame: pd.DataFrame) -> np.ndarray:
    """Return selected source row positions in original CSV order."""
    fraud_mask = frame["fraud"].eq(1)
    legitimate = frame.loc[frame["fraud"].eq(0)]
    target_legitimate_count = TARGET_ROW_COUNT - int(fraud_mask.sum())
    counts_by_step = legitimate.groupby("step", sort=True).size()
    quotas_by_step = _allocate_legitimate_quotas(
        counts_by_step, target_legitimate_count
    )

    rng = np.random.default_rng(RANDOM_SEED)
    legitimate_positions: list[np.ndarray] = []
    for step, group in legitimate.groupby("step", sort=True):
        selected_within_step = rng.choice(
            len(group), size=int(quotas_by_step.loc[step]), replace=False
        )
        legitimate_positions.append(group.index.to_numpy()[selected_within_step])

    fraud_positions = frame.index[fraud_mask].to_numpy()
    selected_positions = np.concatenate([fraud_positions, *legitimate_positions])
    selected_positions.sort()
    return selected_positions


def _validate_source(frame: pd.DataFrame) -> None:
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {EXPECTED_COLUMNS}, found {list(frame.columns)}"
        )
    if frame.isna().any().any():
        raise ValueError("Source CSV contains missing values")
    if not frame["fraud"].isin([0, 1]).all():
        raise ValueError("Source fraud labels must be binary values 0 or 1")
    if len(frame) != EXPECTED_SOURCE_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROW_COUNT:,} source rows, found {len(frame):,}"
        )
    fraud_count = int(frame["fraud"].eq(1).sum())
    if fraud_count != EXPECTED_FRAUD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FRAUD_COUNT:,} fraud rows, found {fraud_count:,}"
        )


def _validate_sample(source: pd.DataFrame, sample: pd.DataFrame) -> None:
    if len(sample) != TARGET_ROW_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_ROW_COUNT:,} sample rows, found {len(sample):,}"
        )

    fraud_count = int(sample["fraud"].eq(1).sum())
    legitimate_count = int(sample["fraud"].eq(0).sum())
    if fraud_count != EXPECTED_FRAUD_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAUD_COUNT:,} sample fraud rows, found {fraud_count:,}"
        )
    expected_legitimate_count = TARGET_ROW_COUNT - EXPECTED_FRAUD_COUNT
    if legitimate_count != expected_legitimate_count:
        raise RuntimeError(
            f"Expected {expected_legitimate_count:,} legitimate rows, "
            f"found {legitimate_count:,}"
        )

    source_steps = set(source["step"].unique())
    sample_steps = set(sample["step"].unique())
    if sample_steps != source_steps:
        missing_steps = sorted(source_steps - sample_steps)
        raise RuntimeError(f"Sample does not cover all source steps: {missing_steps}")

    source_legitimate_steps = set(
        source.loc[source["fraud"].eq(0), "step"].unique()
    )
    sample_legitimate_steps = set(
        sample.loc[sample["fraud"].eq(0), "step"].unique()
    )
    if sample_legitimate_steps != source_legitimate_steps:
        missing_legitimate_steps = sorted(
            source_legitimate_steps - sample_legitimate_steps
        )
        raise RuntimeError(
            "Sample has no legitimate row for source steps that contain "
            f"legitimate transactions: {missing_legitimate_steps}"
        )

    prevalence = fraud_count / len(sample) * 100
    expected_prevalence = EXPECTED_FRAUD_COUNT / TARGET_ROW_COUNT * 100
    if not np.isclose(prevalence, expected_prevalence):
        raise RuntimeError(
            f"Expected sample fraud prevalence {expected_prevalence:.6f}%, "
            f"found {prevalence:.6f}%"
        )
    print(f"Sample rows: {len(sample):,}")
    print(f"Fraud rows: {fraud_count:,}")
    print(f"Legitimate rows: {legitimate_count:,}")
    print(f"Sample fraud prevalence: {prevalence:.6f}%")
    print(
        f"Step coverage: {len(sample_steps):,}/{len(source_steps):,} "
        f"(steps {min(sample_steps)} through {max(sample_steps)})"
    )
    print(
        "Legitimate-row step coverage: "
        f"{len(sample_legitimate_steps):,}/{len(source_legitimate_steps):,}"
    )


def main() -> None:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"BankSim CSV not found: {SOURCE_CSV}")

    source = pd.read_csv(SOURCE_CSV)
    _validate_source(source)

    selected_positions = _select_row_positions(source)
    repeated_positions = _select_row_positions(source)
    if not np.array_equal(selected_positions, repeated_positions):
        raise RuntimeError("Sampling is not reproducible with the configured seed")
    print(f"Reproducibility check: passed (seed={RANDOM_SEED})")
    sample = source.iloc[selected_positions].copy()
    _validate_sample(source, sample)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_CSV, index=False)
    saved_sample = pd.read_csv(OUTPUT_CSV)
    _validate_sample(source, saved_sample)
    pd.testing.assert_frame_equal(
        sample.reset_index(drop=True),
        saved_sample,
        check_dtype=False,
        check_exact=False,
    )

    source_prevalence = EXPECTED_FRAUD_COUNT / EXPECTED_SOURCE_ROW_COUNT * 100
    print(f"Saved sample: {OUTPUT_CSV}")
    print(f"Sampling seed: {RANDOM_SEED}")
    print(f"Original population fraud prevalence: {source_prevalence:.6f}%")
    print(
        "Warning: sampled fraud prevalence differs from the source population; "
        "do not use this sample as the final unbiased test set."
    )


if __name__ == "__main__":
    main()
