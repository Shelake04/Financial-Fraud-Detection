"""Explore the prepared BankSim development sample and save a report and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_object_dtype, is_string_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "processed" / "banksim_sample.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "eda_report.md"
EXPECTED_ROWS = 50_000
SAMPLE_FRAUD_PREVALENCE_PERCENT = 14.4
ORIGINAL_FRAUD_PREVALENCE_PERCENT = 1.210811
MIN_ENTITY_TRANSACTIONS = 20
TOP_ENTITY_COUNT = 10
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
NUMERIC_COLUMNS = {"step", "amount", "fraud"}
CATEGORICAL_COLUMNS = set(EXPECTED_COLUMNS) - NUMERIC_COLUMNS


def _validate_dataset(data: pd.DataFrame) -> None:
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected columns. Expected {EXPECTED_COLUMNS}, "
            f"found {data.columns.tolist()}"
        )
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows, found {len(data):,}")

    for column in ("step", "fraud"):
        if not is_integer_dtype(data[column].dtype):
            raise TypeError(
                f"Expected integer dtype for {column}, got {data[column].dtype}"
            )
    if not is_numeric_dtype(data["amount"].dtype):
        raise TypeError(
            f"Expected numeric dtype for amount, got {data['amount'].dtype}"
        )
    for column in CATEGORICAL_COLUMNS:
        if not (
            is_object_dtype(data[column].dtype)
            or is_string_dtype(data[column].dtype)
        ):
            raise TypeError(
                f"Expected string/object dtype for {column}, "
                f"got {data[column].dtype}"
            )
    if not data["fraud"].isin([0, 1]).all():
        raise ValueError("Fraud labels must contain only 0 and 1")

    fraud_counts = data["fraud"].value_counts()
    observed_fraud = int(fraud_counts.get(1, 0))
    observed_legitimate = int(fraud_counts.get(0, 0))
    if observed_fraud != 7_200 or observed_legitimate != 42_800:
        raise ValueError(
            "Unexpected class counts: "
            f"fraud={observed_fraud:,}, legitimate={observed_legitimate:,}"
        )


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "(missing)"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    frame: pd.DataFrame,
    formatters: dict[str, Callable[[object], str]] | None = None,
) -> str:
    formatters = formatters or {}
    headers = [str(column) for column in frame.columns]
    rows = [
        [
            formatters.get(column, _display_value)(value)
            for column, value in row.items()
        ]
        for _, row in frame.iterrows()
    ]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _group_summary(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = (
        data.groupby(group_column, dropna=False, observed=True)
        .agg(
            transactions=("fraud", "size"),
            fraud_count=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
        )
        .reset_index()
    )
    grouped["fraud_rate_percent"] = grouped["fraud_rate"] * 100
    return grouped


def _age_sort_key(value: object) -> tuple[int, int | str]:
    label = _display_value(value)
    unquoted_label = label.strip("'\"")
    try:
        return (0, int(unquoted_label))
    except ValueError:
        return (1, label)


def _save_amount_figure(data: pd.DataFrame) -> None:
    plot_data = pd.DataFrame(
        {
            "fraud": data["fraud"].map({0: "Legitimate (0)", 1: "Fraud (1)"}),
            "log1p_amount": np.log1p(data["amount"]),
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_data,
        x="fraud",
        y="log1p_amount",
        order=["Legitimate (0)", "Fraud (1)"],
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Transaction Amount by Fraud Label")
    ax.set_xlabel("Transaction class")
    ax.set_ylabel("log(1 + amount); outliers hidden for readability")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "amount_by_fraud.png", dpi=160)
    plt.close(fig)


def _save_category_figure(category_summary: pd.DataFrame) -> None:
    ordered = category_summary.sort_values(
        "fraud_rate_percent", ascending=True, kind="stable"
    )
    fig, (count_ax, rate_ax) = plt.subplots(
        ncols=2, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 1.2]}
    )
    sns.barplot(
        data=ordered,
        x="transactions",
        y="category_label",
        color="#4C78A8",
        ax=count_ax,
    )
    sns.barplot(
        data=ordered,
        x="fraud_rate_percent",
        y="category_label",
        color="#E45756",
        ax=rate_ax,
    )
    count_ax.set_title("Transaction Counts")
    count_ax.set_xlabel("Transactions")
    count_ax.set_ylabel("Category")
    rate_ax.set_title("Fraud Rate")
    rate_ax.set_xlabel("Fraud rate (%)")
    rate_ax.set_ylabel("")
    fig.suptitle("Transaction Counts and Fraud Rates by Category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "category_fraud_patterns.png", dpi=160)
    plt.close(fig)


def _save_step_figure(step_summary: pd.DataFrame) -> None:
    fig, (count_ax, rate_ax) = plt.subplots(
        nrows=2, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    count_ax.bar(step_summary["step"], step_summary["transactions"], color="#4C78A8")
    count_ax.set_title("Transactions per Step")
    count_ax.set_ylabel("Transactions")
    rate_ax.plot(
        step_summary["step"],
        step_summary["fraud_rate_percent"],
        color="#E45756",
        linewidth=1.5,
    )
    rate_ax.set_title("Fraud Rate per Step")
    rate_ax.set_xlabel("Step")
    rate_ax.set_ylabel("Fraud rate (%)")
    rate_ax.set_xlim(step_summary["step"].min(), step_summary["step"].max())
    fig.suptitle("Transaction Volume and Fraud Rate over Steps")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_rate_by_step.png", dpi=160)
    plt.close(fig)


def _save_age_figure(age_summary: pd.DataFrame) -> None:
    fig, (count_ax, rate_ax) = plt.subplots(
        ncols=2, figsize=(12, 5), sharey=True, gridspec_kw={"width_ratios": [1, 1.2]}
    )
    sns.barplot(
        data=age_summary,
        x="transactions",
        y="age_label",
        color="#4C78A8",
        ax=count_ax,
    )
    sns.barplot(
        data=age_summary,
        x="fraud_rate_percent",
        y="age_label",
        color="#E45756",
        ax=rate_ax,
    )
    count_ax.set_title("Transaction Counts")
    count_ax.set_xlabel("Transactions")
    count_ax.set_ylabel("Recorded age value")
    rate_ax.set_title("Fraud Rate")
    rate_ax.set_xlabel("Fraud rate (%)")
    rate_ax.set_ylabel("")
    fig.suptitle("Transaction Counts and Fraud Rates by Recorded Age Value")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fraud_rate_by_age_value.png", dpi=160)
    plt.close(fig)


def _entity_summary(data: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    grouped = (
        data.groupby(entity_column, dropna=False, observed=True)
        .agg(
            transactions=("fraud", "size"),
            fraud_count=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
        )
        .reset_index()
    )
    grouped["entity_label"] = grouped[entity_column].map(_display_value)
    grouped["fraud_rate_percent"] = grouped["fraud_rate"] * 100
    return grouped


def _save_entity_figure(
    entity_summaries: dict[str, pd.DataFrame],
    sample_fraud_rate_percent: float,
) -> None:
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6))
    for ax, (entity_name, summary) in zip(axes, entity_summaries.items()):
        supported = summary.loc[
            summary["transactions"] >= MIN_ENTITY_TRANSACTIONS
        ]
        ax.scatter(
            supported["transactions"],
            supported["fraud_rate_percent"],
            alpha=0.65,
            s=24,
        )
        ax.axhline(
            sample_fraud_rate_percent,
            color="#E45756",
            linestyle="--",
            label="Sample-wide fraud rate",
        )
        ax.set_title(f"{entity_name.title()} patterns")
        ax.set_xlabel("Transactions per entity")
        ax.set_ylabel("Fraud rate (%)")
        ax.legend()
    fig.suptitle(
        "Entity Transaction Volume and Fraud Rate "
        f"(entities with at least {MIN_ENTITY_TRANSACTIONS} transactions)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "customer_merchant_patterns.png", dpi=160)
    plt.close(fig)


def _entity_report_sections(
    entity_summaries: dict[str, pd.DataFrame],
) -> list[str]:
    sections: list[str] = []
    for entity_name, summary in entity_summaries.items():
        top_volume = summary.sort_values(
            ["transactions", "fraud_rate_percent", "entity_label"],
            ascending=[False, False, True],
            kind="stable",
        ).head(TOP_ENTITY_COUNT)
        high_rate = (
            summary.loc[summary["transactions"] >= MIN_ENTITY_TRANSACTIONS]
            .sort_values(
                ["fraud_rate_percent", "fraud_count", "transactions"],
                ascending=[False, False, False],
                kind="stable",
            )
            .head(TOP_ENTITY_COUNT)
        )
        columns = [
            "entity_label",
            "transactions",
            "fraud_count",
            "fraud_rate_percent",
        ]
        percent = {"fraud_rate_percent": lambda value: f"{value:.3f}%"}
        sections.extend(
            [
                f"### {entity_name.title()}",
                "",
                "Highest transaction volume (volume is context, not evidence of fraud):",
                "",
                _markdown_table(top_volume[columns], percent),
                "",
                f"Highest fraud rates with at least {MIN_ENTITY_TRANSACTIONS} "
                "transactions:",
                "",
                _markdown_table(high_rate[columns], percent),
                "",
            ]
        )
    return sections


def _build_report(
    data: pd.DataFrame,
    missing_counts: pd.Series,
    duplicate_count: int,
    amount_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    step_summary: pd.DataFrame,
    age_summary: pd.DataFrame,
    entity_summaries: dict[str, pd.DataFrame],
) -> str:
    fraud_counts = data["fraud"].value_counts()
    fraud_count = int(fraud_counts.get(1, 0))
    legitimate_count = int(fraud_counts.get(0, 0))
    fraud_rate_percent = data["fraud"].mean() * 100
    step_min = int(data["step"].min())
    step_max = int(data["step"].max())
    step_count = int(data["step"].nunique())
    nonzero_missing = missing_counts[missing_counts.gt(0)]
    missing_text = (
        _markdown_table(nonzero_missing.rename("missing_values").rename_axis("column").reset_index())
        if not nonzero_missing.empty
        else "No missing values were found."
    )

    sections = [
        "# Exploratory Data Analysis: BankSim Development Sample",
        "",
        "## Scope and prevalence",
        "",
        f"This analysis uses only `{SAMPLE_CSV.relative_to(PROJECT_ROOT)}`. "
        "It does not read the original source dataset.",
        "",
        f"The sample contains **{len(data):,} rows** and **{len(data.columns)} "
        f"original columns**, with **{fraud_count:,} fraud** and "
        f"**{legitimate_count:,} legitimate** transactions. Its fraud prevalence "
        f"is **{fraud_rate_percent:.6f}%** (expected sample baseline: "
        f"{SAMPLE_FRAUD_PREVALENCE_PERCENT:.6f}%), compared with "
        f"**{ORIGINAL_FRAUD_PREVALENCE_PERCENT:.6f}%** in the original dataset.",
        "",
        "Because this sample retains every fraud transaction while downsampling "
        "legitimate transactions, its prevalence and group fraud rates are "
        "descriptive of this enriched sample, not unbiased estimates of the "
        "original population.",
        "",
        "## Data quality and structure",
        "",
        f"- Shape: {len(data):,} rows × {len(data.columns)} columns.",
        f"- Exact duplicate rows: {duplicate_count:,}.",
        f"- Step coverage: {step_count} observed values, from {step_min} to {step_max}.",
        f"- Fraud rate: {fraud_rate_percent:.6f}%.",
        "- Missing values by column:",
        "",
        missing_text,
        "",
        "## Transaction amounts by fraud label",
        "",
        _markdown_table(
            amount_summary,
            {
                "mean": lambda value: f"{value:.3f}",
                "median": lambda value: f"{value:.3f}",
                "std": lambda value: f"{value:.3f}",
                "min": lambda value: f"{value:.3f}",
                "max": lambda value: f"{value:.3f}",
            },
        ),
        "",
        "Figure: `../figures/amount_by_fraud.png`. The box plot uses log(1 + amount) "
        "to improve readability and omits plotted outlier markers; summary "
        "statistics use the original amounts.",
        "",
        "## Category patterns",
        "",
        _markdown_table(
            category_summary[
                ["category_label", "transactions", "fraud_count", "fraud_rate_percent"]
            ],
            {"fraud_rate_percent": lambda value: f"{value:.3f}%"},
        ),
        "",
        "Compare both transaction counts and fraud rates; rates are sample-based "
        "and should be interpreted with their denominators.",
        "",
        "Figure: `../figures/category_fraud_patterns.png`.",
        "",
        "## Fraud rates by step",
        "",
        f"Step-level transaction counts and fraud rates cover all {step_count} "
        "observed step values. Small per-step denominators can make rates "
        "variable; the chart shows counts alongside rates. Highest observed "
        "step rates are listed with their transaction counts:",
        "",
        _markdown_table(
            step_summary.sort_values(
                ["fraud_rate_percent", "transactions"],
                ascending=[False, False],
                kind="stable",
            )
            .head(TOP_ENTITY_COUNT)[
                ["step", "transactions", "fraud_count", "fraud_rate_percent"]
            ],
            {"fraud_rate_percent": lambda value: f"{value:.3f}%"},
        ),
        "",
        "Figure: `../figures/fraud_rate_by_step.png`.",
        "",
        "## Fraud rates by recorded age value",
        "",
        "Age is analyzed using the literal categorical values present in the "
        "dataset, without assigning unsupported demographic meanings to the codes.",
        "",
        _markdown_table(
            age_summary[
                ["age_label", "transactions", "fraud_count", "fraud_rate_percent"]
            ],
            {"fraud_rate_percent": lambda value: f"{value:.3f}%"},
        ),
        "",
        "Figure: `../figures/fraud_rate_by_age_value.png`.",
        "",
        "## Customer and merchant patterns",
        "",
        "Transaction volume is descriptive context, not evidence of fraud. "
        "Tables show both counts and fraud rates; rate rankings require at least "
        f"{MIN_ENTITY_TRANSACTIONS} transactions per entity to avoid highlighting "
        "rates based on very small denominators. All results remain specific to "
        "the fraud-enriched sample.",
        "",
    ]
    sections.extend(_entity_report_sections(entity_summaries))
    sections.extend(
        [
            "Figure: `../figures/customer_merchant_patterns.png`. Each point "
            "represents an entity with the stated minimum support; the dashed "
            "line marks the sample-wide fraud rate.",
            "",
            "## Figures",
            "",
            "- `../figures/amount_by_fraud.png`",
            "- `../figures/category_fraud_patterns.png`",
            "- `../figures/fraud_rate_by_step.png`",
            "- `../figures/fraud_rate_by_age_value.png`",
            "- `../figures/customer_merchant_patterns.png`",
            "",
        ]
    )
    return "\n".join(sections)


def main() -> None:
    if not SAMPLE_CSV.is_file():
        raise FileNotFoundError(f"Processed sample not found: {SAMPLE_CSV}")

    data = pd.read_csv(SAMPLE_CSV)
    _validate_dataset(data)

    missing_counts = data.isna().sum()
    duplicate_count = int(data.duplicated().sum())
    fraud_counts = data["fraud"].value_counts()
    fraud_rate_percent = data["fraud"].mean() * 100

    amount_summary = (
        data.groupby("fraud", observed=True)["amount"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    amount_summary["fraud_label"] = amount_summary["fraud"].map(
        {0: "Legitimate (0)", 1: "Fraud (1)"}
    )
    amount_summary = amount_summary[
        ["fraud_label", "count", "mean", "median", "std", "min", "max"]
    ]

    category_summary = _group_summary(data, "category")
    category_summary["category_label"] = category_summary["category"].map(_display_value)
    category_summary = category_summary.sort_values(
        ["fraud_rate_percent", "transactions"], ascending=[False, False], kind="stable"
    )

    step_summary = (
        data.groupby("step", observed=True)
        .agg(
            transactions=("fraud", "size"),
            fraud_count=("fraud", "sum"),
            fraud_rate=("fraud", "mean"),
        )
        .reset_index()
        .sort_values("step")
    )
    step_summary["fraud_rate_percent"] = step_summary["fraud_rate"] * 100

    age_summary = _group_summary(data, "age")
    age_summary["age_label"] = age_summary["age"].map(_display_value)
    age_summary["_sort_key"] = age_summary["age"].map(_age_sort_key)
    age_summary = age_summary.sort_values("_sort_key", kind="stable").drop(
        columns="_sort_key"
    )

    entity_summaries = {
        entity_name: _entity_summary(data, entity_name)
        for entity_name in ("customer", "merchant")
    }

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    _save_amount_figure(data)
    _save_category_figure(category_summary)
    _save_step_figure(step_summary)
    _save_age_figure(age_summary)
    _save_entity_figure(entity_summaries, fraud_rate_percent)

    report = _build_report(
        data,
        missing_counts,
        duplicate_count,
        amount_summary,
        category_summary,
        step_summary,
        age_summary,
        entity_summaries,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Dataset shape: {len(data):,} rows × {len(data.columns)} columns")
    print(f"Missing values: {int(missing_counts.sum()):,}")
    print(f"Exact duplicate rows: {duplicate_count:,}")
    print(
        f"Class counts: legitimate={int(fraud_counts.get(0, 0)):,}, "
        f"fraud={int(fraud_counts.get(1, 0)):,}"
    )
    print(
        f"Fraud prevalence: {fraud_rate_percent:.6f}% in sample "
        f"(original dataset: {ORIGINAL_FRAUD_PREVALENCE_PERCENT:.6f}%)"
    )
    print(f"Step coverage: {data['step'].nunique()} observed values")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Figures saved under: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
