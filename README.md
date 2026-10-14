# Financial Fraud Detection Using Machine Learning

A machine learning-based financial fraud detection system built using Python, Pandas, Scikit-learn, XGBoost, and Streamlit.

## Dashboard Preview

The Streamlit dashboard provides an interactive interface for exploring transaction fraud patterns, reviewing model evaluation results, and demonstrating fraud-risk predictions.

### Overview

![Dashboard Overview](overview.png)

### Fraud Analysis

![Fraud Analysis](fraud-analysis.png)

### Model Evaluation

![Model Evaluation](model-evaluation.png)

### Fraud Prediction

![Fraud Prediction](fraud-prediction.png)

## Project Status

The project analyzes financial transaction patterns, engineers customer and merchant behavioral features, compares machine learning classification models, and provides an interactive dashboard for exploring fraud patterns and demonstrating predictions.

> **Project status:** The core machine learning pipeline, prediction service, automated tests, and interactive dashboard have been implemented. Operational fraud prediction is not enabled.

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Project Workflow](#project-workflow)
- [Feature Engineering](#feature-engineering)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Prediction Service](#prediction-service)
- [Interactive Dashboard](#interactive-dashboard)
- [Automated Testing](#automated-testing)
- [Installation and Execution](#installation-and-execution)
- [Project Limitations](#project-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Financial fraud detection is a classification problem in which transaction data is analyzed to identify potentially fraudulent activity.

This project uses the BankSim dataset to explore transaction patterns, create behavioral features from historical customer and merchant activity, train classification models, and evaluate their ability to distinguish fraudulent transactions from legitimate ones.

The project also includes a prediction service and a Streamlit dashboard for data exploration, model evaluation, and synthetic prediction demonstrations.

### Key Features

- BankSim dataset inspection and sampling
- Exploratory Data Analysis (EDA)
- Customer and merchant behavioral feature engineering
- Chronological training, validation, and test splits
- Data preprocessing
- Logistic Regression, Random Forest, and XGBoost model comparison
- Validation-based model and decision-threshold selection
- Final test evaluation
- Historical transaction feature store
- Model prediction service
- Interactive Streamlit dashboard
- Automated testing with Pytest

## Problem Statement

Financial transaction fraud can cause financial losses for customers and financial institutions. Analyzing transaction characteristics and historical behavioral patterns can help identify suspicious activity.

The objective of this project is to develop and evaluate a machine learning pipeline that classifies financial transactions using transaction attributes and behavioral information derived from previous customer and merchant activity.

## Objectives

- Analyze financial transaction data.
- Explore differences between legitimate and fraudulent transactions.
- Identify transaction categories and behavioral patterns associated with fraud in the development sample.
- Engineer behavioral features using historical transaction activity.
- Compare multiple machine learning classification algorithms.
- Evaluate models using appropriate classification metrics.
- Build a reusable prediction service.
- Develop an interactive dashboard for data analysis and model evaluation.
- Demonstrate model predictions using synthetic transaction inputs.

## Dataset

This project uses the BankSim dataset.

### Original Dataset

| Description | Count |
|---|---:|
| Total transactions | 594,643 |
| Fraudulent transactions | 7,200 |
| Legitimate transactions | 587,443 |
| Fraud prevalence | Approximately 1.21% |

### Development Sample

A 50,000-row development sample was created for experimentation.

| Description | Count |
|---|---:|
| Total sampled transactions | 50,000 |
| Fraudulent transactions | 7,200 |
| Legitimate transactions | 42,800 |
| Fraud prevalence | 14.4% |

The sampling procedure retained all 7,200 fraudulent transactions and selected 42,800 legitimate transactions. Sampling used a fixed random seed and preserved the chronological step structure.

**Important:** The development sample has a substantially higher fraud prevalence than the original dataset. Precision, fraud rates, and other evaluation results obtained from this enriched sample may not represent performance on naturally occurring financial transactions.

The original dataset is not included in the Git repository. Users must obtain a compatible dataset separately.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, classification utilities, and evaluation |
| XGBoost | Gradient-boosted classification |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Plotly | Interactive visualization |
| Streamlit | Interactive dashboard |
| Joblib | Saving and loading trained models and preprocessing artifacts |
| Pytest | Automated testing |
| Git | Version control |
| GitHub | Source-code hosting |

## Project Structure

The repository is organized into the following main directories:

```text
Financial-Fraud-Detection/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── 01_data_inspection.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
├── src/
│   ├── create_sample.py
│   ├── eda.py
│   ├── evaluate_model.py
│   ├── feature_engineering.py
│   ├── feature_store.py
│   ├── inspect_dataset.py
│   ├── prediction_service.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   └── train_model.py
├── tests/
│   ├── test_feature_store.py
│   ├── test_prediction_dashboard.py
│   └── test_prediction_service.py
├── .gitignore
├── README.md
└── requirements.txt
```

Generated datasets, trained model artifacts, and other local outputs are excluded from version control.

## Project Workflow

The project follows this workflow:

1. Inspect the original dataset.
2. Create a development sample.
3. Perform Exploratory Data Analysis.
4. Engineer customer and merchant behavioral features.
5. Split the data chronologically into training, validation, and test sets.
6. Fit the preprocessing pipeline using training data.
7. Train Logistic Regression, Random Forest, and XGBoost.
8. Compare models using validation metrics.
9. Select the final model and decision threshold using validation data.
10. Evaluate the selected model on the held-out test set.
11. Develop the historical feature store and prediction service.
12. Integrate the prediction service and model artifacts with the Streamlit dashboard.
13. Run automated tests.

## Feature Engineering

The project combines transaction attributes with historical customer and merchant behavior.

### Behavioral Features

Examples include:

- Customer's previous transaction count
- Customer's previous transaction amount total
- Customer's previous transaction amount mean
- Steps since the customer's previous activity
- Merchant's previous transaction count
- Merchant's previous transaction amount total
- Merchant's previous transaction amount mean
- Steps since the merchant's previous activity

### Prevention of Temporal Leakage

Historical behavioral features are calculated using strictly earlier transaction steps.

Transactions belonging to the same step use the same pre-step history snapshot. Historical state is updated only after the entire step has been processed.

Customer and merchant identifiers are used to derive behavioral features rather than being passed directly to the model as input features.

Fraud labels are not used to construct behavioral history.

The preprocessing pipeline is fitted on training data and then applied to validation and test data.

## Machine Learning Models

Three classification algorithms were trained and compared.

### 1. Logistic Regression

Logistic Regression provides a baseline classification model for estimating the probability that a transaction is fraudulent.

### 2. Random Forest

Random Forest combines multiple decision trees to model nonlinear relationships between transaction attributes and fraud outcomes.

### 3. XGBoost

XGBoost uses gradient-boosted decision trees to learn complex patterns in transaction data.

### Model Selection

XGBoost was selected because it achieved the highest validation Average Precision among the evaluated models.

Model selection and decision-threshold selection were performed using validation data. The held-out test set was evaluated separately.

## Model Evaluation

The selected XGBoost model produced the following results on the held-out test split.

### Test Metrics

| Metric | Result |
|---|---:|
| Average Precision (AP) | 0.992662 |
| ROC-AUC | 0.998882 |
| Precision | 0.938322 |
| Recall | 0.966667 |
| F1-score | 0.952283 |
| Decision threshold | 0.718754 |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 6,360 | 61 |
| Fraud | 32 | 928 |

### Interpretation

On the held-out test split:

- 928 fraudulent transactions were correctly identified.
- 32 fraudulent transactions were missed.
- 61 legitimate transactions were incorrectly flagged as fraudulent.
- 6,360 legitimate transactions were correctly classified.

These results describe performance on the fraud-enriched development sample. They do not establish that equivalent results will be achieved on a naturally distributed transaction population.

The decision threshold reflects the threshold selected during validation and should not be assumed to be optimal for every financial institution or operational setting.

## Prediction Service

The prediction service integrates the saved XGBoost model with the saved preprocessing pipeline and historical behavioral feature store.

Its responsibilities include:

- Validating transaction inputs
- Generating behavioral features
- Applying the fitted preprocessing pipeline
- Calculating fraud probabilities
- Applying the selected decision threshold
- Validating model and preprocessing compatibility
- Managing historical state during prediction

The service commits historical state only after successful scoring and validation.

The feature store is currently in-memory. It does not provide durable storage or thread-safe state management.

## Interactive Dashboard

The project includes a Streamlit dashboard with four pages.

### 1. Overview

Displays project information and a summary of the selected model's evaluation results.

### 2. Fraud Analysis

Presents visualizations covering:

- Transaction amounts
- Fraud patterns by category
- Fraud rates across transaction steps
- Age-code patterns
- Customer and merchant activity

The age values in the BankSim dataset are categorical codes, not actual numerical ages.

### 3. Model Evaluation

Displays validation metrics, final test metrics, and model comparison information using the generated reports.

This page displays existing evaluation results; it does not independently retrain or reevaluate the model.

### 4. Fraud Prediction

Provides a synthetic transaction demonstration using the saved model and preprocessing artifacts.

Users can enter transaction attributes and observe the resulting probability and threshold-based classification.

**The prediction interface is a demonstration, not an operational fraud detection service.** It uses fabricated inputs and empty in-memory behavioral history. Consequently, its predictions should not be interpreted as decisions about real financial transactions.

Operational prediction remains disabled because a trusted transaction history source, persistent state management, and additional production safeguards have not been implemented.

## Automated Testing

The project contains automated tests for:

- Historical behavioral feature generation
- Transaction history state management
- Prediction service validation
- Preprocessing and model compatibility
- Probability and classification outputs
- Dashboard input validation
- Dashboard error handling

**87 tests passed.**

The reported result represents the test run performed during development.
Run the test suite again after making changes to verify the current state
of the project.

Run the tests from the repository root:

```bash
python -m pytest tests/ -v
```

The reported result represents the test run performed during development. Run the test suite again after making changes to verify the current state of the project.

## Installation and Execution

### Prerequisites

- Python installed on your computer
- Git
- A compatible BankSim dataset obtained separately

### 1. Clone the Repository

```bash
git clone https://github.com/Shelake04/Financial-Fraud-Detection.git
cd Financial-Fraud-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Prepare the Dataset and Artifacts

The prediction service and dashboard require the saved XGBoost model and preprocessing artifacts. These generated files are excluded from the repository.

Obtain a compatible BankSim dataset and place it in the location expected by the data preparation scripts.

Run the project's data inspection, sample creation, preprocessing, and model training scripts in the appropriate order. These steps generate the data and model artifacts required by the dashboard.

The model evaluation script can be run separately to evaluate the trained model on the prepared test dataset.

**Note:** Confirm the expected dataset path and script execution order in the source code before running the pipeline.

### 5. Run the Dashboard

From the repository root, run:

```bash
python -m streamlit run app/app.py
```

Open the local address displayed in the terminal.

The dashboard requires the necessary prepared-data, preprocessing, and model artifacts to have been generated first.

## Project Limitations

- The development sample has an artificially elevated fraud prevalence.
- Evaluation results may not generalize to naturally occurring financial transactions.
- The dashboard prediction demonstration uses fabricated transaction data.
- The demonstration begins with empty in-memory behavioral history.
- The feature store is not persistent or thread-safe.
- A trusted, production-quality historical transaction source has not been integrated.
- Production monitoring, audit logging, and operational safeguards remain future work.
- The application is not ready for operational fraud decisions.

## Future Improvements

- Evaluate performance on a dataset with a representative fraud prevalence.
- Integrate a trusted transaction history source.
- Implement persistent and concurrency-safe behavioral state management.
- Add data-quality monitoring.
- Add model-performance monitoring and drift detection.
- Implement prediction auditing and operational safeguards.
- Establish a reproducible model training and deployment workflow.
- Explore Apache Kafka and PySpark Structured Streaming for transaction processing.

## Author

## Dashboard Preview

The Streamlit dashboard provides an interactive interface for exploring transaction fraud patterns, reviewing model evaluation results, and demonstrating fraud-risk predictions.

### Overview

![Dashboard Overview](overview.png)

### Fraud Analysis

![Fraud Analysis](fraud-analysis.png)

### Model Evaluation

![Model Evaluation](model-evaluation.png)

### Fraud Prediction

![Fraud Prediction](fraud-prediction.png)

## Project Status

The project analyzes financial transaction patterns, engineers customer and merchant behavioral features, compares machine learning classification models, and provides an interactive dashboard for exploring fraud patterns and demonstrating predictions.