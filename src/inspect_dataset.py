"""Inspect the external BankSim CSV without loading it all into memory."""

from __future__ import annotations

import json
import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CSV = Path(r"D:\Fraud Detection\bs140513_032310.csv")
CHUNK_SIZE = 50_000
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


def _canonical_row(values: tuple[object, ...]) -> str:
    normalized = []
    for value in values:
        if pd.isna(value):
            normalized.append(None)
        elif hasattr(value, "item"):
            normalized.append(value.item())
        else:
            normalized.append(value)
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _update_numeric_stats(stats: dict[str, float | int | None], values: pd.Series) -> None:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if not numeric.size:
        return

    batch_count = int(numeric.size)
    batch_mean = float(numeric.mean())
    batch_m2 = float(np.square(numeric - batch_mean).sum())
    old_count = int(stats["count"])
    old_mean = float(stats["mean"]) if old_count else 0.0
    delta = batch_mean - old_mean
    new_count = old_count + batch_count

    stats["m2"] = float(stats["m2"]) + batch_m2 + delta * delta * old_count * batch_count / new_count
    stats["mean"] = old_mean + delta * batch_count / new_count
    stats["count"] = new_count
    stats["min"] = min(float(stats["min"]) if stats["min"] is not None else math.inf, float(numeric.min()))
    stats["max"] = max(float(stats["max"]) if stats["max"] is not None else -math.inf, float(numeric.max()))


def _print_numeric_stats(name: str, stats: dict[str, float | int | None]) -> None:
    count = int(stats["count"])
    if not count:
        print(f"{name}: no valid numeric values")
        return
    variance = float(stats["m2"]) / (count - 1) if count > 1 else float("nan")
    print(
        f"{name}: count={count:,}, mean={float(stats['mean']):.6g}, "
        f"std={math.sqrt(variance):.6g}, min={float(stats['min']):.6g}, "
        f"max={float(stats['max']):.6g}"
    )


def main() -> None:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"BankSim CSV not found: {SOURCE_CSV}")

    columns = pd.read_csv(SOURCE_CSV, nrows=0).columns.tolist()
    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {EXPECTED_COLUMNS}, found {columns}"
        )

    row_count = 0
    missing_counts = {column: 0 for column in columns}
    observed_dtypes = {column: set() for column in columns}
    fraud_count = 0
    non_fraud_count = 0
    other_fraud_values = 0
    numeric_stats = {
        column: {"count": 0, "mean": 0.0, "m2": 0.0, "min": None, "max": None}
        for column in ("amount", "step")
    }

    with tempfile.TemporaryDirectory(prefix="banksim-exact-duplicates-") as temp_dir:
        database_path = Path(temp_dir) / "rows.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.execute(
                "CREATE TABLE unique_rows (row_json TEXT PRIMARY KEY) WITHOUT ROWID"
            )

            for chunk in pd.read_csv(SOURCE_CSV, chunksize=CHUNK_SIZE):
                row_count += len(chunk)
                for column in columns:
                    missing_counts[column] += int(chunk[column].isna().sum())
                    observed_dtypes[column].add(str(chunk[column].dtype))

                fraud_values = pd.to_numeric(chunk["fraud"], errors="coerce")
                fraud_count += int(fraud_values.eq(1).sum())
                non_fraud_count += int(fraud_values.eq(0).sum())
                other_fraud_values += int((~fraud_values.isin([0, 1])).sum())

                for column in numeric_stats:
                    _update_numeric_stats(numeric_stats[column], chunk[column])

                serialized_rows = (
                    (_canonical_row(row),) for row in chunk.itertuples(index=False, name=None)
                )
                connection.executemany(
                    "INSERT OR IGNORE INTO unique_rows (row_json) VALUES (?)",
                    serialized_rows,
                )
                connection.commit()

            unique_count = int(
                connection.execute("SELECT COUNT(*) FROM unique_rows").fetchone()[0]
            )
        finally:
            connection.close()

    duplicate_count = row_count - unique_count
    print(f"Source: {SOURCE_CSV}")
    print(f"Total rows: {row_count:,}")
    print("Columns and observed pandas dtypes:")
    for column in columns:
        print(f"  {column}: {', '.join(sorted(observed_dtypes[column]))}")
    print("Missing values per column:")
    for column in columns:
        print(f"  {column}: {missing_counts[column]:,}")
    print(f"Exact duplicate rows: {duplicate_count:,}")
    print("Fraud label counts (percent of all rows):")
    print(f"  fraud (1): {fraud_count:,} ({fraud_count / row_count * 100:.6f}%)")
    print(f"  non-fraud (0): {non_fraud_count:,} ({non_fraud_count / row_count * 100:.6f}%)")
    if other_fraud_values:
        print(f"  missing or non-binary fraud labels: {other_fraud_values:,}")
    print("Basic numeric statistics (sample standard deviation):")
    _print_numeric_stats("amount", numeric_stats["amount"])
    _print_numeric_stats("step", numeric_stats["step"])


if __name__ == "__main__":
    main()