"""Streamlit foundation for the Financial Fraud Detection dashboard."""

from __future__ import annotations

from collections.abc import Callable
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
import src.prediction_service as prediction_service


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
VALIDATION_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "model_validation_report.md"
)
TEST_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "reports" / "test_evaluation_report.md"
)
SYNTHETIC_HISTORY_COMPLETE_THROUGH = 0
SYNTHETIC_DEMO_STEP = SYNTHETIC_HISTORY_COMPLETE_THROUGH + 1
SYNTHETIC_HISTORY_COLUMNS = ("step", "amount", "customer", "merchant")
SYNTHETIC_TRANSACTION_COLUMNS = (
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
)
PROJECT_TITLE = "Financial Fraud Detection"
PROJECT_DESCRIPTION = (
    "An analytical dashboard for exploring transaction fraud patterns and "
    "supporting future fraud-risk review."
)
TEST_RESULTS = {
    "rows": 7_381,
    "actual_fraud": 960,
    "predicted_fraud": 989,
    "fraud_prevalence_percent": 13.006368,
    "threshold": 0.71875411,
    "precision": 0.938322,
    "recall": 0.966667,
    "f1": 0.952283,
    "average_precision_ap": 0.992662,
    "roc_auc": 0.998882,
    "confusion_matrix": [[6_360, 61], [32, 928]],
}
PROJECT_WORKFLOW = [
    "Data inspection and sampling",
    "Exploratory data analysis",
    "Behavioral feature engineering",
    "Model training and validation",
    "Held-out test evaluation",
]


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f7fb;
            color: #172b3f;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {
            background-color: #f4f7fb;
            color: #172b3f;
        }
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
        [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
            color: #172b3f;
        }
        [data-testid="stSidebar"] {
            background-color: #10243a;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
        [data-testid="stSidebar"] [role="radiogroup"],
        [data-testid="stSidebar"] [role="radiogroup"] *,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #edf4fb !important;
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="false"] {
            color: #d6e2ee !important;
        }
        .dashboard-eyebrow {
            color: #087381;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .dashboard-subtitle {
            color: #40566c;
            font-size: 1.08rem;
            max-width: 52rem;
        }
        [data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #d4dfe9;
            border-radius: 0.75rem;
            padding: 1rem;
            color: #172b3f;
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"] {
            color: #172b3f !important;
        }
        [data-testid="stAlert"] {
            background-color: #eaf2f8;
            border: 1px solid #bfd0df;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
            color: #172b3f !important;
        }
        [data-testid="stAlert"][kind="warning"] {
            background-color: #fff4d6;
            border-color: #e7c66b;
        }
        [data-testid="stAlert"][kind="error"] {
            background-color: #fdeceb;
            border-color: #e4aaa5;
        }
        [data-testid="stAlert"][kind="success"] {
            background-color: #e7f4ec;
            border-color: #a7cfb4;
        }
        [data-testid="stCaptionContainer"] {
            color: #40566c !important;
        }
        [data-testid="stTable"],
        [data-testid="stDataFrame"] {
            color: #172b3f;
        }
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            color: #172b3f !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        '<p class="dashboard-eyebrow">Risk Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.title(PROJECT_TITLE)
    st.markdown(
        f'<p class="dashboard-subtitle">{PROJECT_DESCRIPTION}</p>',
        unsafe_allow_html=True,
    )


def _render_overview() -> None:
    st.subheader("Project overview")
    st.write(
        "A reproducible fraud-detection workflow built around BankSim "
        "transactions, chronological data preparation, and validation-led "
        "model selection."
    )

    kpi_columns = st.columns(4)
    kpi_values = [
        ("Development sample", "50,000", "transactions"),
        ("Selected model", "XGBoost", "chosen by validation AP"),
        ("Test Average Precision (AP)", "0.992662", "held-out test"),
        ("Test ROC-AUC", "0.998882", "held-out test"),
    ]
    for column, (label, value, help_text) in zip(kpi_columns, kpi_values):
        column.metric(label, value, help=help_text)

    st.warning(
        "The 50,000-row development sample is fraud-enriched: every fraud record "
        "was retained while legitimate records were sampled. Its 14.4% fraud "
        "prevalence is higher than the original dataset's 1.210811%. Rates and "
        "metrics from this sample should not be treated as population estimates."
    )

    left_column, right_column = st.columns([1.1, 0.9], gap="large")
    with left_column:
        st.markdown("### Project workflow")
        for step_number, workflow_step in enumerate(PROJECT_WORKFLOW, start=1):
            st.markdown(f"**{step_number}.** {workflow_step}")
        st.caption(
            "Behavioral features use strictly earlier transaction steps; model "
            "selection and threshold choice use validation data."
        )

    with right_column:
        st.markdown("### Held-out test snapshot")
        st.caption(
            "Reported from the completed evaluation using the saved XGBoost "
            "model and its validation-selected threshold."
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric("Precision", f"{TEST_RESULTS['precision']:.6f}")
        metric_columns[1].metric("Recall", f"{TEST_RESULTS['recall']:.6f}")
        metric_columns[2].metric("F1-score", f"{TEST_RESULTS['f1']:.6f}")
        st.write(
            f"**Transactions:** {TEST_RESULTS['rows']:,}  \n"
            f"**Actual fraud:** {TEST_RESULTS['actual_fraud']:,}  \n"
            f"**Predicted fraud:** {TEST_RESULTS['predicted_fraud']:,}  \n"
            f"**Fraud prevalence:** "
            f"{TEST_RESULTS['fraud_prevalence_percent']:.6f}%  \n"
            f"**Applied validation threshold:** "
            f"{TEST_RESULTS['threshold']:.8f}"
        )

    st.markdown("#### Confusion matrix")
    st.caption("Rows are actual labels; columns are predicted labels.")
    confusion_columns = st.columns([1, 2, 2])
    confusion_columns[0].markdown("**Actual class**")
    confusion_columns[1].markdown("**Predicted legitimate**")
    confusion_columns[2].markdown("**Predicted fraud**")
    confusion_columns[0].markdown("Legitimate")
    confusion_columns[1].write(f"{TEST_RESULTS['confusion_matrix'][0][0]:,}")
    confusion_columns[2].write(f"{TEST_RESULTS['confusion_matrix'][0][1]:,}")
    confusion_columns[0].markdown("Fraud")
    confusion_columns[1].write(f"{TEST_RESULTS['confusion_matrix'][1][0]:,}")
    confusion_columns[2].write(f"{TEST_RESULTS['confusion_matrix'][1][1]:,}")

    st.info(
        "These held-out test metrics describe the prepared, fraud-enriched "
        "sample. They do not guarantee performance on transactions in the "
        "real world, where prevalence, behavior, and data quality may differ."
    )


def _render_fraud_analysis() -> None:
    st.subheader("Fraud Analysis")
    st.write(
        "Explore transaction amounts, category and step-level patterns, "
        "recorded age codes, and customer/merchant behavior."
    )
    st.warning(
        "These figures describe the fraud-enriched 50,000-row development "
        "sample (14.4% fraud), not the original population. Sample-based "
        "category and group fraud rates may not represent population rates."
    )

    figure_rows = [
        [
            (
                "Transaction amounts by fraud label",
                "amount_by_fraud.png",
                "The amount distributions are shown by transaction label; "
                "the EDA plot uses log(1 + amount) for readability.",
            ),
            (
                "Fraud patterns by category",
                "category_fraud_patterns.png",
                "Compare category fraud rates alongside transaction counts; "
                "rates are specific to this sampled dataset.",
            ),
        ],
        [
            (
                "Fraud rate across time steps",
                "fraud_rate_by_step.png",
                "Step-level rates are displayed with transaction counts so "
                "their denominators remain visible.",
            ),
            (
                "Fraud rate by recorded age value",
                "fraud_rate_by_age_value.png",
                "Age values are dataset category codes, not verified actual "
                "ages or demographic descriptions.",
            ),
        ],
    ]
    for figure_row in figure_rows:
        columns = st.columns(2, gap="large")
        for column, (heading, filename, description) in zip(columns, figure_row):
            with column:
                st.markdown(f"### {heading}")
                st.caption(description)
                _display_eda_figure(filename, heading)

    st.markdown("### Customer and merchant patterns")
    st.caption(
        "Transaction volume is descriptive context, not evidence of fraud. "
        "Interpret entity fraud rates with their transaction counts and the "
        "sample-enrichment caveat."
    )
    _display_eda_figure(
        "customer_merchant_patterns.png",
        "Customer and merchant patterns",
    )


def _display_eda_figure(filename: str, alt_text: str) -> None:
    figure_path = FIGURES_DIR / filename
    if not figure_path.is_file():
        st.error(
            f"EDA figure is missing: `{figure_path.relative_to(PROJECT_ROOT)}`. "
            "Restore the existing figure to display it."
        )
        return
    st.image(str(figure_path), caption=alt_text, use_container_width=True)


def _render_fraud_prediction() -> None:
    st.subheader("Fraud Prediction")
    st.warning(
        "**Synthetic demonstration only.** This uses fabricated, empty history "
        "and is not an operational fraud assessment."
    )

    demo_inputs: dict[str, Any] = {}
    with st.form("transaction_input_form"):
        st.markdown("### Transaction details")
        first_row = st.columns(3)
        demo_inputs["step"] = first_row[0].number_input(
            "Step",
            min_value=0,
            value=SYNTHETIC_DEMO_STEP,
            step=1,
            key="demo_step",
            help="The synthetic history fixture supports step 1 only.",
        )
        demo_inputs["amount"] = first_row[1].number_input(
            "Amount",
            min_value=0.0,
            step=1.0,
            format="%.2f",
            key="demo_amount",
        )
        demo_inputs["age"] = first_row[2].text_input(
            "Age code",
            help="Enter the dataset's recorded age category code, not an interpreted age.",
            key="demo_age",
        )

        second_row = st.columns(3)
        demo_inputs["gender"] = second_row[0].text_input(
            "Gender code", key="demo_gender"
        )
        demo_inputs["category"] = second_row[1].text_input(
            "Transaction category", key="demo_category"
        )
        demo_inputs["zipcodeOri"] = second_row[2].text_input(
            "Origin ZIP code", key="demo_zipcode_ori"
        )

        third_row = st.columns(3)
        demo_inputs["zipMerchant"] = third_row[0].text_input(
            "Merchant ZIP code", key="demo_zip_merchant"
        )
        demo_inputs["customer"] = third_row[1].text_input(
            "Customer ID", key="demo_customer"
        )
        demo_inputs["merchant"] = third_row[2].text_input(
            "Merchant ID", key="demo_merchant"
        )

        st.caption(
            "Raw transaction fields captured: step, amount, age, gender, "
            "category, zipcodeOri, zipMerchant, customer, and merchant."
        )
        run_demo = st.form_submit_button(
            "Run synthetic demonstration",
            key="run_synthetic_demo",
            type="primary",
            use_container_width=True,
        )
        st.form_submit_button(
            "Operational prediction unavailable",
            key="operational_prediction_unavailable",
            disabled=True,
            use_container_width=True,
        )

    if run_demo:
        try:
            transaction = _build_synthetic_demo_transaction(demo_inputs)
        except (TypeError, ValueError) as error:
            st.error(f"Input validation error: {error}")
        else:
            synthetic_history = pd.DataFrame(
                {
                    "step": pd.Series(dtype="int64"),
                    "amount": pd.Series(dtype="float64"),
                    "customer": pd.Series(dtype="object"),
                    "merchant": pd.Series(dtype="object"),
                },
                columns=SYNTHETIC_HISTORY_COLUMNS,
            )
            try:
                service = prediction_service.PredictionService(
                    synthetic_history,
                    history_complete_through=SYNTHETIC_HISTORY_COMPLETE_THROUGH,
                )
            except Exception as error:
                st.error(
                    "Synthetic demonstration could not load or validate the "
                    f"model artifacts: {error}"
                )
            else:
                try:
                    results = service.predict(transaction)
                    probability, threshold, predicted_fraud = (
                        _validate_synthetic_demo_result(
                            results,
                            transaction,
                            service.threshold,
                        )
                    )
                except Exception as error:
                    st.error(f"Synthetic demonstration scoring error: {error}")
                else:
                    st.markdown("### Synthetic demonstration result")
                    st.caption(
                        "Demonstration only; this result is not an operational "
                        "fraud assessment."
                    )
                    st.write(
                        f"**Transaction ID:** `{transaction.iloc[0]['transaction_id']}`"
                    )
                    result_columns = st.columns(3)
                    result_columns[0].metric("Fraud probability", f"{probability:.6f}")
                    result_columns[1].metric("Stored threshold", f"{threshold:.6f}")
                    result_columns[2].metric(
                        "Threshold-based classification",
                        "Fraud" if predicted_fraud else "Not flagged",
                    )

    st.markdown("### Required behavioral history")
    st.write(
        "The trained model also requires customer and merchant prior "
        "transaction counts, previous amount totals and means, and steps since "
        "each entity's previous activity. Customer and merchant IDs are used "
        "to derive this history; they are not model input features themselves."
    )
    st.warning(
        "A valid prediction requires trusted customer and merchant history "
        "from strictly earlier transaction steps. Transactions sharing a step "
        "must all use the same pre-step history; update history only after all "
        "transactions at that step are processed."
    )
    st.error(
        "The fraud-enriched development sample is not a suitable operational "
        "history source: legitimate transactions were downsampled, so entity "
        "counts and spending summaries may not represent complete activity."
    )
    st.caption(
        "Fraud is the target label, not a transaction input, and must never be "
        "used to construct behavioral history."
    )
    st.info(
        "The synthetic demonstration creates a fresh service with empty "
        "in-memory history complete through step 0 and scores step 1 only. "
        "The operational prediction action remains disabled until trustworthy "
        "operational history and durable, concurrency-safe state management "
        "are implemented."
    )


def _build_synthetic_demo_transaction(inputs: dict[str, Any]) -> pd.DataFrame:
    step = inputs["step"]
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("Step must be an integer.")
    if step != SYNTHETIC_DEMO_STEP:
        raise ValueError(
            f"The synthetic history fixture supports step {SYNTHETIC_DEMO_STEP} only."
        )

    try:
        amount = float(inputs["amount"])
    except (TypeError, ValueError) as error:
        raise ValueError("Amount must be a finite, non-negative number.") from error
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a finite, non-negative number.")

    transaction: dict[str, Any] = {"step": step, "amount": amount}
    for field in (
        "age",
        "gender",
        "category",
        "zipcodeOri",
        "zipMerchant",
        "customer",
        "merchant",
    ):
        value = inputs[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty value.")
        transaction[field] = value.strip()
    transaction["transaction_id"] = f"synthetic-{uuid4()}"

    return pd.DataFrame(
        [[transaction[column] for column in SYNTHETIC_TRANSACTION_COLUMNS]],
        columns=SYNTHETIC_TRANSACTION_COLUMNS,
    )


def _validate_synthetic_demo_result(
    results: pd.DataFrame,
    transaction: pd.DataFrame,
    stored_threshold: float,
) -> tuple[float, float, bool]:
    expected_columns = {
        "transaction_id",
        "step",
        "fraud_probability",
        "threshold",
        "predicted_fraud",
    }
    if (
        not isinstance(results, pd.DataFrame)
        or len(results) != 1
        or not expected_columns.issubset(results.columns)
        or not results.index.equals(transaction.index)
        or results["transaction_id"].tolist()
        != transaction["transaction_id"].tolist()
        or results["step"].tolist() != transaction["step"].tolist()
    ):
        raise ValueError("Prediction results do not align with the submitted transaction.")

    probability = float(results.iloc[0]["fraud_probability"])
    threshold = float(results.iloc[0]["threshold"])
    if (
        not math.isfinite(probability)
        or not 0 <= probability <= 1
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
        or threshold != float(stored_threshold)
    ):
        raise ValueError("Prediction returned an invalid probability or threshold.")
    predicted_fraud = results.iloc[0]["predicted_fraud"]
    if not isinstance(predicted_fraud, (bool, np.bool_)):
        raise ValueError("Prediction returned an invalid fraud classification.")
    predicted_fraud = bool(predicted_fraud)
    if predicted_fraud != (probability >= threshold):
        raise ValueError("Prediction classification does not match its threshold.")
    return probability, threshold, predicted_fraud


def _render_model_evaluation() -> None:
    st.subheader("Model Evaluation")
    st.write(
        "Validation results compare candidate models and select the model and "
        "decision threshold. The held-out test results are a final assessment "
        "of the selected model using that unchanged threshold."
    )

    validation_report = _read_evaluation_report(
        VALIDATION_REPORT_PATH, "validation"
    )
    test_report = _read_evaluation_report(TEST_REPORT_PATH, "held-out test")
    if validation_report is None or test_report is None:
        return

    validation_results = _parse_validation_report(validation_report)
    test_results = _parse_test_report(test_report)
    if validation_results is None or test_results is None:
        return

    st.markdown("### Validation comparison")
    validation_rows = [
        [
            result["model"],
            f"{result['threshold']:.8f}",
            f"{result['precision']:.6f}",
            f"{result['recall']:.6f}",
            f"{result['f1']:.6f}",
            f"{result['average_precision_ap']:.6f}",
            f"{result['roc_auc']:.6f}",
        ]
        for result in validation_results
    ]
    st.table(
        {
            "Model": [row[0] for row in validation_rows],
            "Threshold": [row[1] for row in validation_rows],
            "Precision": [row[2] for row in validation_rows],
            "Recall": [row[3] for row in validation_rows],
            "F1": [row[4] for row in validation_rows],
            "Average Precision (AP)": [row[5] for row in validation_rows],
            "ROC-AUC": [row[6] for row in validation_rows],
        }
    )
    selected_model = max(
        validation_results, key=lambda result: result["average_precision_ap"]
    )
    st.success(
        f"Selected candidate: **{selected_model['model']}**, with the highest "
        "validation Average Precision (AP) "
        f"({selected_model['average_precision_ap']:.6f})."
    )

    st.markdown("### Final held-out test results")
    st.caption(
        "These results come from the existing test evaluation report. The test "
        "set was not used for model selection or threshold tuning."
    )
    test_metric_columns = st.columns(5)
    test_metrics = [
        ("Average Precision (AP)", test_results["average_precision_ap"]),
        ("ROC-AUC", test_results["roc_auc"]),
        ("Precision", test_results["precision"]),
        ("Recall", test_results["recall"]),
        ("F1-score", test_results["f1"]),
    ]
    for column, (metric_name, metric_value) in zip(
        test_metric_columns, test_metrics
    ):
        column.metric(metric_name, f"{metric_value:.6f}")

    st.metric(
        "Validation-selected threshold",
        f"{test_results['threshold']:.8f}",
        help="Chosen using validation data and reused unchanged for the held-out test.",
    )
    st.markdown("#### Confusion matrix")
    st.caption("Rows are actual classes; columns are predicted classes.")
    confusion = test_results["confusion_matrix"]
    st.table(
        {
            "Actual class": ["Legitimate", "Fraud"],
            "Predicted legitimate": [
                f"{confusion[0][0]:,}",
                f"{confusion[1][0]:,}",
            ],
            "Predicted fraud": [
                f"{confusion[0][1]:,}",
                f"{confusion[1][1]:,}",
            ],
        }
    )

    st.warning(
        "The held-out test partition comes from the fraud-enriched development "
        "sample. Its reported fraud prevalence is "
        f"{test_results['fraud_prevalence_percent']:.6f}%, compared with "
        "1.210811% in the original dataset. Test results describe this prepared "
        "sample and may not represent original-population or real-world "
        "performance."
    )


def _read_evaluation_report(path: Path, report_name: str) -> str | None:
    if not path.is_file():
        st.error(
            f"The {report_name} report is missing: "
            f"`{path.relative_to(PROJECT_ROOT)}`."
        )
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        st.error(f"Could not read the {report_name} report: {error}")
        return None


def _parse_validation_report(report: str) -> list[dict[str, Any]] | None:
    row_pattern = re.compile(
        r"^\| (Logistic Regression|Random Forest|XGBoost) "
        r"\| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) "
        r"\| ([0-9.]+) \| ([0-9.]+) \|",
        re.MULTILINE,
    )
    results = [
        {
            "model": match.group(1),
            "threshold": float(match.group(2)),
            "precision": float(match.group(3)),
            "recall": float(match.group(4)),
            "f1": float(match.group(5)),
            "average_precision_ap": float(match.group(6)),
            "roc_auc": float(match.group(7)),
        }
        for match in row_pattern.finditer(report)
    ]
    expected_models = {"Logistic Regression", "Random Forest", "XGBoost"}
    if {result["model"] for result in results} != expected_models:
        st.error(
            "The validation report is missing expected model metrics or has an "
            "unrecognized table format."
        )
        return None
    return results


def _parse_test_report(report: str) -> dict[str, Any] | None:
    metric_names = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "average_precision_ap": "Average Precision (AP)",
        "roc_auc": "ROC-AUC",
    }
    results: dict[str, Any] = {}
    for key, label in metric_names.items():
        match = re.search(
            rf"^\| {re.escape(label)} \| ([0-9.]+) \|$",
            report,
            re.MULTILINE,
        )
        if match is None:
            st.error(f"The held-out test report is missing the {label} metric.")
            return None
        results[key] = float(match.group(1))

    summary_patterns = {
        "threshold": r"^- Applied validation-selected threshold: ([0-9.]+)$",
        "fraud_prevalence_percent": r"^- Fraud prevalence: ([0-9.]+)%$",
    }
    for key, pattern in summary_patterns.items():
        match = re.search(pattern, report, re.MULTILINE)
        if match is None:
            st.error(
                "The held-out test report is missing its saved threshold or "
                "fraud-prevalence summary."
            )
            return None
        results[key] = float(match.group(1))

    confusion_pattern = re.compile(
        r"^- Legitimate row: `\[(\d+),\s*(\d+)\]`\s*\n"
        r"^- Fraud row: `\[(\d+),\s*(\d+)\]`",
        re.MULTILINE,
    )
    confusion_match = confusion_pattern.search(report)
    if confusion_match is None:
        st.error("The held-out test report is missing its labeled confusion matrix.")
        return None
    results["confusion_matrix"] = [
        [int(confusion_match.group(1)), int(confusion_match.group(2))],
        [int(confusion_match.group(3)), int(confusion_match.group(4))],
    ]
    return results


def main() -> None:
    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_theme()

    with st.sidebar:
        st.markdown("### Risk Intelligence")
        st.caption("BankSim fraud analytics")
        selected_page = st.radio(
            "Dashboard navigation",
            options=[
                "Overview",
                "Fraud Analysis",
                "Fraud Prediction",
                "Model Evaluation",
            ],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Financial Fraud Detection | Dashboard foundation")

    _render_header()
    st.divider()

    page_renderers: dict[str, Callable[[], None]] = {
        "Overview": _render_overview,
        "Fraud Analysis": _render_fraud_analysis,
        "Fraud Prediction": _render_fraud_prediction,
        "Model Evaluation": _render_model_evaluation,
    }
    renderer = page_renderers.get(selected_page)
    if renderer is None:
        st.error(f"Unknown dashboard section: {selected_page}")
        return
    renderer()


if __name__ == "__main__":
    main()
