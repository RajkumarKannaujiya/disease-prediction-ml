"""Generate dataset cards, the project report and viva guide from real outputs."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from disease_prediction.data import DATASET_KEYS, load_dataset  # noqa: E402
from disease_prediction.schemas import DISEASES  # noqa: E402


FEATURE_NOTES = {
    "diabetes": {
        "Pregnancies": "Number of recorded pregnancies.",
        "Glucose": "Plasma glucose concentration from the 2-hour oral glucose tolerance test.",
        "BloodPressure": "Diastolic blood pressure in mm Hg.",
        "SkinThickness": "Triceps skinfold thickness in millimetres.",
        "Insulin": "2-hour serum insulin measurement.",
        "BMI": "Body mass index.",
        "DiabetesPedigreeFunction": "Dataset pedigree-function score related to family history.",
        "Age": "Age in years.",
    },
    "heart": {
        "age": "Age in years.", "sex": "Dataset sex code (0/1).", "cp": "Chest-pain category code.",
        "trestbps": "Resting blood pressure in mm Hg.", "chol": "Serum cholesterol in mg/dL.",
        "fbs": "Indicator for fasting blood sugar above the source threshold.",
        "restecg": "Resting electrocardiographic category code.", "thalach": "Maximum heart rate achieved.",
        "exang": "Exercise-induced angina indicator.", "oldpeak": "ST depression induced by exercise relative to rest.",
        "slope": "Slope category of the peak exercise ST segment.",
        "ca": "Number of major vessels colored by fluoroscopy.", "thal": "Thalassemia category code.",
    },
    "liver": {
        "age": "Age in years.", "gender": "Source gender category.",
        "total_bilirubin": "Total bilirubin measurement.", "direct_bilirubin": "Direct bilirubin measurement.",
        "alkphos": "Alkaline phosphatase measurement.", "sgpt": "SGPT / alanine aminotransferase measurement.",
        "sgot": "SGOT / aspartate aminotransferase measurement.", "total_proteins": "Total protein measurement.",
        "albumin": "Albumin measurement.", "ag_ratio": "Albumin-to-globulin ratio.",
    },
    "kidney": {
        "age": "Age in years.", "bp": "Blood pressure in mm Hg.", "sg": "Urine specific-gravity category.",
        "al": "Urine albumin category.", "su": "Urine sugar category.", "rbc": "Red-blood-cell test category.",
        "pc": "Pus-cell test category.", "pcc": "Pus-cell-clumps category.", "ba": "Bacteria category.",
        "bgr": "Random blood glucose measurement.", "bu": "Blood urea measurement.",
        "sc": "Serum creatinine measurement.", "sod": "Sodium measurement.", "pot": "Potassium measurement.",
        "hemo": "Hemoglobin measurement.", "pcv": "Packed cell volume.", "wc": "White blood cell count.",
        "rc": "Red blood cell count.", "htn": "Hypertension history category.",
        "dm": "Diabetes mellitus history category.", "cad": "Coronary artery disease history category.",
        "appet": "Appetite category.", "pe": "Pedal-edema category.", "ane": "Anemia category.",
    },
}


VIVA = [
    ("Why did you choose this project?", "It combines data provenance, preprocessing, comparative classification, evaluation, explainability, a web interface, and privacy-aware history. The goal is a reproducible academic demonstration, not a clinical product."),
    ("What is machine learning?", "Machine learning is a set of methods that estimate patterns from examples and apply those learned patterns to new inputs."),
    ("Why supervised learning?", "Each selected dataset contains input variables and a defined target label, so the task is supervised classification."),
    ("Why classification?", "The source targets represent two classes after documented binary mapping, such as the CKD versus non-CKD label."),
    ("What is a feature?", "A feature is an input column used by a model. In this project each form is limited to fields present in that module's training data."),
    ("What is a target variable?", "The target is the label the model learns to classify. Its definition comes from the source dataset and is not interchangeable across modules."),
    ("What is data leakage?", "Leakage occurs when information unavailable at prediction time, or information from evaluation data, influences model fitting or selection."),
    ("How did you reduce leakage?", "Imputation, encoding, and scaling are inside an sklearn Pipeline. Hyperparameter selection uses training folds, and exact duplicate feature rows are grouped into one split."),
    ("Why detect duplicate records?", "Duplicates can inflate measured performance if copies cross train and test partitions. Identical feature rows are grouped instead of automatically removed because they may represent distinct records."),
    ("Why use grouped splitting?", "It prevents identical feature vectors from appearing in both training and evaluation partitions, reducing an obvious source of optimistic leakage."),
    ("What is a Pipeline?", "An sklearn Pipeline chains preprocessing and an estimator so each cross-validation fold fits transformations only on its training portion."),
    ("Why impute missing values?", "Many estimators do not accept missing values. The numeric median and categorical most-frequent value are learned from training data only."),
    ("Why are some Pima zeros treated as missing?", "The source documentation identifies zero-coded values for several measurements as physically implausible missing placeholders. Pregnancies=0 remains a valid value."),
    ("Why encode categorical fields?", "Most estimators need numeric arrays. One-hot encoding represents nominal categories without inventing an ordinal distance."),
    ("Why scale numeric features?", "Scaling helps distance- and margin-based models such as KNN and SVM, and can improve optimization for Logistic Regression."),
    ("What is Logistic Regression?", "It models the log-odds of a class as a linear function of inputs and can provide a useful, comparatively interpretable baseline."),
    ("How does a Decision Tree work?", "It recursively partitions the feature space using rules chosen to reduce impurity, producing an interpretable but potentially high-variance model."),
    ("Why use Random Forest?", "It averages many bootstrapped trees with randomized feature selection, often reducing variance compared with one tree."),
    ("What is an SVM?", "A Support Vector Machine seeks a separating boundary with a large margin; kernels allow nonlinear boundaries."),
    ("How does KNN classify?", "KNN assigns a class using nearby training points. It depends on feature scaling and can be sensitive to distance and sample representation."),
    ("Why evaluate XGBoost?", "It is a gradient-boosted tree candidate for tabular classification. It is included as a comparison, not assumed to be best."),
    ("What is cross-validation?", "Cross-validation rotates held-out folds within training data to estimate how choices vary across different training subsets."),
    ("What is hyperparameter tuning?", "It searches model settings such as tree depth or regularization strength. Here it uses GridSearchCV inside grouped training folds."),
    ("Why use a fixed random seed?", "A fixed seed makes data splitting and stochastic model setup repeatable for the same data and software environment."),
    ("What is stratification?", "Stratification aims to preserve class proportions across splits, which is useful when positive and negative counts differ."),
    ("What is accuracy?", "Accuracy is the fraction of all records classified correctly. It can look high while an important minority class is missed."),
    ("What is precision?", "Precision is the share of predicted positives that are true positives; low precision means more false alarms."),
    ("What is recall?", "Recall or sensitivity is the share of actual positives detected; low recall means more false negatives."),
    ("Why is recall considered for medical-risk projects?", "Missing a positive label can be consequential, so sensitivity must be examined. Increasing sensitivity may also increase false positives."),
    ("What is the F1 score?", "F1 is the harmonic mean of precision and recall, balancing the two when both matter."),
    ("What is ROC-AUC?", "ROC-AUC summarizes ranking across thresholds using true-positive and false-positive rates. It does not select a safe operating threshold."),
    ("What is average precision?", "Average precision summarizes precision-recall performance and is useful when class prevalence is uneven."),
    ("What does a confusion matrix show?", "It counts true negatives, false positives, false negatives, and true positives for the selected threshold."),
    ("What is a threshold?", "It converts a score or probability estimate into a class decision. A threshold reflects trade-offs and should be chosen using training/validation data, not the final test set."),
    ("What is probability calibration?", "A calibrated probability has empirical outcome frequencies close to the stated probability over comparable cases. This project does not claim clinical calibration."),
    ("What is a Brier score?", "It is mean squared error between predicted probabilities and binary outcomes; lower is better, but it is affected by prevalence and calibration."),
    ("How did you address class imbalance?", "Candidate classifiers use class weights where supported, and performance reports include recall and precision-recall metrics. No synthetic rows enter the test set."),
    ("What is overfitting?", "Overfitting occurs when a model learns noise or sample-specific patterns that do not generalize to new records."),
    ("What is underfitting?", "Underfitting occurs when a model is too simple or constrained to capture useful patterns in the training data."),
    ("What is feature selection?", "Feature selection chooses a subset of input variables. It must be fitted inside training folds to avoid using validation/test outcomes."),
    ("What is SHAP?", "SHAP assigns additive feature contributions relative to a reference distribution for a specific model output."),
    ("Does SHAP prove a feature causes disease?", "No. SHAP explains model behavior under a chosen background; it is not causal inference or a medical conclusion."),
    ("Why use model-agnostic SHAP here?", "The same permutation approach can explain different selected estimators. One-hot contributions are grouped back to the original dataset variables."),
    ("How does Flask serve predictions?", "Flask routes accept requests, validate form values, call a prediction service that loads a saved pipeline, and render a result page."),
    ("Why use SQLite?", "SQLite is a small local relational database with no separate server process, suitable for a single-user academic demonstration."),
    ("How are SQL injections reduced?", "Database writes and reads use parameterized SQL values instead of concatenating user input into statements."),
    ("Why add CSRF protection?", "CSRF tokens help ensure state-changing form submissions originated from the application session."),
    ("How are models saved?", "Joblib serializes each fitted preprocessing-and-estimator pipeline; JSON metadata records its version, features, source hash, and evaluation summary."),
    ("What is the risk of Joblib files?", "Joblib uses pickle-based serialization, which can execute code when loading malicious artifacts. The app must load only trusted project-generated files."),
    ("What happens during a prediction?", "The server validates one input row, applies the saved transformations, predicts a dataset class and score, computes local contributions when possible, and stores the minimal history entry."),
    ("Why does the application store no name?", "The demo only needs module, fields, output, timestamp, and model version for its history feature; names and contact details are unnecessary."),
    ("Why can the system not replace a doctor?", "It has no clinical validation, uses narrow historical datasets, may be biased or miscalibrated, and cannot interpret an individual's full clinical context."),
    ("Why isn't there a general symptom checker?", "The selected modules use different clinical or survey variables. A single symptom model would need a suitable multi-disease dataset and independent validation."),
    ("What is external validation?", "It evaluates a frozen model on a separate population or site that was not used in training or model selection."),
    ("What is population shift?", "Population shift occurs when feature distributions, prevalence, measurement practices, or relationships differ between training data and deployment users."),
    ("Why are the four module probabilities not comparable?", "The modules have different cohorts, target labels, class prevalence, and modeling procedures, so the numeric outputs have different meanings."),
    ("What does the model version identify?", "It identifies the selected trained artifact version and makes history records traceable to a reproducible pipeline manifest."),
    ("What is the difference between a global and local explanation?", "A global summary describes contributions across a sample; a local explanation describes contributions for one submitted record relative to a reference."),
]

ADVANCED_VIVA = [
    ("Why does the holdout score remain valid if the saved artifact is later refit on all rows?", "The reported holdout estimate was computed before the refit using a model selected only on training data. The full-data artifact is for deployment/demo after evaluation; its performance is not independently measured on data it has seen. External validation is still required."),
    ("How would you select a probability threshold without contaminating the test set?", "Generate out-of-fold probabilities within development data, select a threshold using a predeclared objective or cost trade-off, lock it, then evaluate once on the untouched test set."),
    ("Why can very high CKD test metrics still be untrustworthy?", "The test set is small and drawn from the same benchmark. Strong features may reflect the source diagnosis process, while population shift, label leakage, and sampling variation remain possible."),
    ("How does grouped CV reduce duplicate leakage when no patient ID exists?", "Hash identical feature rows into groups and ensure each group stays in one fold. It prevents exact-vector copies crossing partitions but cannot identify different records from the same person."),
    ("What if missingness is not at random?", "Median or mode imputation can be biased if missingness depends on the unobserved value or disease state. Analyze missingness patterns, report them, and validate alternative approaches without fitting on held-out data."),
    ("Why is recall-based model selection not automatically clinically safe?", "Recall ignores false-positive burden, calibration, operational consequences, and clinical utility. A threshold and model require stakeholder-defined costs and prospective validation."),
    ("How do you aggregate one-hot SHAP values?", "Explain the transformed estimator inputs, then sum encoded-category contributions belonging to each original variable. This preserves the additive contribution across the encodings."),
    ("Why keep synthetic oversampling out of the full pipeline?", "If oversampling happens before splitting, near-copies can leak into validation/test data. If justified, it must be fitted only inside each training fold and evaluated on the original class distribution."),
    ("What is the difference between ranking quality and calibration?", "ROC-AUC measures ranking across thresholds; calibration measures agreement between predicted probability and observed frequency. A model can rank well and be poorly calibrated."),
    ("What evidence would be needed before real clinical deployment?", "Clear intended-use definition, representative external and prospective validation, calibration and threshold evaluation, subgroup/fairness analysis, clinical workflow review, security/privacy controls, and applicable regulatory review."),
]


def write_dataset_sources(results: dict, source_manifest: list[dict]) -> str:
    source_hashes = {item["dataset_key"]: item for item in source_manifest}
    lines = ["# Dataset sources, fields, and measured quality", "", f"Generated: {date.today().isoformat()}", ""]
    for key in DATASET_KEYS:
        disease = DISEASES[key]
        result = results[key]
        quality = result["data_quality"]
        source_item = source_hashes.get(key, {})
        lines.extend([
            f"## {disease.title}", "",
            f"- **Dataset:** {disease.dataset_name}",
            f"- **Source:** {disease.dataset_source}",
            f"- **Source citation/license:** {disease.dataset_license}",
            f"- **Target:** {disease.target_description}",
            f"- **Source records:** {quality['source_records']}; **modeling rows:** {quality['modeling_records']}; **exact duplicate feature/target rows:** {quality['exact_duplicate_rows_detected']} (kept and grouped for splitting).",
            f"- **Observed target counts (0/1):** `{quality['target_counts']}`.",
            f"- **Source file SHA-256:** `{source_item.get('sha256', 'not recorded')}`.",
            f"- **Population limitation:** {disease.population_limit}", "",
            "### Features", "", "| Feature | Meaning | Type | Missing values | IQR screening flags |", "|---|---|---|---:|---:|",
        ])
        for feature in quality["features"]:
            definition = FEATURE_NOTES[key].get(feature, feature.replace("_", " ").capitalize())
            lines.append(f"| `{feature}` | {definition} | {quality['feature_types'][feature]} | {quality['missing_by_feature'][feature]} | {quality['outlier_iqr_counts'].get(feature, 0)} |")
        lines.extend([
            "", "### Preprocessing plan actually applied", "",
            "- Source-specific target mapping; raw input schema checked against the selected dataset columns.",
            "- Exact duplicate detection; rows are retained and identical feature vectors are grouped into outer and cross-validation partitions.",
            "- Pima zero-coded Glucose, BloodPressure, SkinThickness, Insulin, and BMI values are treated as missing; Pregnancies=0 remains valid.",
            "- Numeric missing values use training-fold median imputation and scaling; categorical missing values use training-fold most-frequent imputation and one-hot encoding.",
            "- IQR flags are descriptive. No automatic outlier removal, target-informed selection, or test-set preprocessing is applied.",
            "- Train/test is grouped and stratified with a fixed seed; all tuning is restricted to grouped CV within training data.", "",
        ])
    return "\n".join(lines)


def measured_results_table(results: dict) -> str:
    lines = ["| Module | Selected model | Accuracy | Precision | Recall / sensitivity | F1 | ROC-AUC | Average precision | Brier |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, result in results.items():
        metric = result["holdout_metrics"]
        lines.append(
            f"| {result['title']} | {result['selected_model']} | {metric['accuracy']:.3f} | {metric['precision']:.3f} | {metric['recall_sensitivity']:.3f} | {metric['f1']:.3f} | {metric['roc_auc']:.3f} | {metric['average_precision']:.3f} | {metric['brier_score']:.3f} |"
        )
    return "\n".join(lines)


def eda_observations() -> str:
    """Summarize measured class balance, missingness, and IQR flags."""
    quality_path = ROOT / "data" / "processed" / "dataset_quality.json"
    quality_by_key = json.loads(quality_path.read_text(encoding="utf-8"))
    paragraphs = []
    for key in DATASET_KEYS:
        quality = quality_by_key[key]
        counts = quality["target_counts"]
        count_text = ", ".join(f"class {label}: {count}" for label, count in sorted(counts.items()))
        missing = sorted(
            ((name, count) for name, count in quality["missing_by_feature"].items() if count),
            key=lambda item: item[1], reverse=True,
        )[:2]
        outliers = sorted(
            ((name, count) for name, count in quality["outlier_iqr_counts"].items() if count),
            key=lambda item: item[1], reverse=True,
        )[:2]
        missing_text = ", ".join(f"`{name}` ({count})" for name, count in missing) or "none observed"
        outlier_text = ", ".join(f"`{name}` ({count})" for name, count in outliers) or "none flagged"
        duplicate_count = quality["exact_duplicate_rows_detected"]
        paragraphs.append(
            f"**{DISEASES[key].title}:** {quality['modeling_records']} records; target counts {count_text}. "
            f"Largest feature missing counts: {missing_text}. Highest IQR screening counts: {outlier_text}. "
            f"Exact duplicate rows detected: {duplicate_count}; duplicates were retained and grouped during splitting."
        )
    return "\n\n".join(paragraphs)


def model_comparison_table(result: dict) -> str:
    lines = ["| Model | CV recall | CV precision | CV F1 | CV ROC-AUC | Test accuracy | Test precision | Test recall | Test F1 | Test ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["model_comparison"]:
        lines.append(
            f"| {row['model']} | {row['cv_recall_sensitivity']:.3f} | {row['cv_precision']:.3f} | {row['cv_f1']:.3f} | {row['cv_roc_auc']:.3f} | {row['test_accuracy']:.3f} | {row['test_precision']:.3f} | {row['test_recall_sensitivity']:.3f} | {row['test_f1']:.3f} | {row['test_roc_auc']:.3f} |"
        )
    return "\n".join(lines)


def test_table() -> tuple[str, str]:
    report_path = ROOT / "reports" / "test-results.xml"
    if not report_path.exists():
        return "Test report not yet generated; run `python -m pytest --junitxml=reports/test-results.xml`.", "Test execution is not yet recorded."
    root = ET.parse(report_path).getroot()
    cases = root.findall(".//testcase")
    failures = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    summary = f"{len(cases) - failures} passed, {failures} failed across {len(cases)} test cases."
    rows = ["| Test ID | Test case | Input | Expected result | Actual result | Status |", "|---|---|---|---|---|---|"]
    mappings = [
        ("T01", "Dataset fields match each form", "Actual downloaded dataset schema", "Exact input field order matches model schema", "test_actual_dataset_columns_match_form_schema"),
        ("T02", "Input validation accepts sourced example rows", "Public-source row with missing fields filled for the fixture", "Valid values returned as a one-row frame", "test_valid_public_record_values_pass_schema"),
        ("T03", "Missing required input", "Omit one required field", "Validation error returned", "test_missing_required_input_is_rejected"),
        ("T04", "Invalid numeric and range input", "Text in numeric field / out-of-range value", "Validation error returned", "test_non_numeric_value_is_rejected"),
        ("T05", "Unknown field is rejected", "Unexpected field in heart module", "Validation error returned", "test_out_of_range_and_unknown_field_are_rejected"),
        ("T06", "Database write, list, and clear", "Parameter-bound record", "Record reads back and deletes cleanly", "test_database_insert_list_and_clear"),
        ("T07", "Model bundle load and prediction", "Sourced diabetes row", "Binary prediction and bounded score returned", "test_model_loading_and_prediction_function"),
        ("T08", "All saved model bundles load", "One sourced row per disease", "All four pipelines predict", "test_every_saved_bundle_loads_and_predicts"),
        ("T09", "Application page routes", "GET on home, about, history, etc.", "HTTP 200", "test_page_routes_return_success"),
        ("T10", "Input forms load", "GET each disease form", "HTTP 200 with module title", "test_each_disease_form_loads"),
        ("T11", "CSRF check", "POST without valid token", "HTTP 400", "test_post_without_csrf_is_rejected"),
        ("T12", "JSON validation route", "Incomplete JSON with valid session token", "HTTP 400 with field errors", "test_json_route_rejects_invalid_payload"),
        ("T13", "Invalid form submission", "Incomplete diabetes form with valid CSRF token", "HTTP 400 with required-field message", "test_invalid_form_submission_returns_validation_message"),
        ("T14", "Prediction route and history integration", "Valid sourced diabetes row with CSRF token", "Prediction page renders and history records model version", "test_valid_prediction_route_records_history"),
        ("T15", "Static favicon asset", "GET /static/favicon.svg", "HTTP 200 with SVG content", "test_favicon_is_served"),
    ]
    names = [case.attrib.get("name", "") for case in cases]
    for test_id, label, input_text, expected, match in mappings:
        matching = [case for case in cases if match in case.attrib.get("name", "")]
        passed = bool(matching) and all(case.find("failure") is None and case.find("error") is None for case in matching)
        rows.append(f"| {test_id} | {label} | {input_text} | {expected} | {'Passed' if passed else 'Failed / not recorded'} | {'PASS' if passed else 'FAIL'} |")
    return "\n".join(rows), summary


def write_report(results: dict) -> str:
    source_manifest = json.loads((ROOT / "data" / "raw" / "source_manifest.json").read_text(encoding="utf-8"))
    test_rows, test_summary = test_table()
    dataset_sources = write_dataset_sources(results, source_manifest)
    (ROOT / "reports" / "DATASET_SOURCES.md").write_text(dataset_sources, encoding="utf-8")
    report = f"""# Project Report: AI-Based Multi-Disease Prediction and Risk Assessment System

**Academic project report · Generated {date.today().isoformat()}**

> Educational and research demonstration only. This work does not provide a medical diagnosis or clinical decision support.

## Chapter 1 — Introduction

### 1.1 Background
Supervised machine learning can learn classification patterns from labeled tabular data. Public benchmark datasets allow reproducible demonstrations of data preparation, model comparison, and evaluation, while their historical and population-specific nature limits how results can be interpreted.

### 1.2 Problem statement
Course projects often report one accuracy value without recording data lineage, leakage controls, feature definitions, or error trade-offs. This project builds an auditable web workflow around four dataset-specific binary classification tasks.

### 1.3 Motivation
The motivation is to demonstrate a complete research workflow that can be explained in a viva: public source acquisition, data auditing, preprocessing, model comparison, class-sensitive evaluation, explainability, and web integration.

### 1.4 Objectives
1. Acquire and cite public datasets. 2. Audit missingness, duplicates, types, distributions, and class balance. 3. Compare multiple classifiers using grouped cross-validation. 4. Evaluate selected models on held-out records. 5. expose dataset-specific inputs and qualified outputs through Flask. 6. Add local SQLite history and reproducible documentation.

### 1.5 Scope
Modules cover diabetes, heart disease, liver disease, and chronic kidney disease. Each task uses only the features present in its source. The system is a research demonstration, not a clinical product.

### 1.6 Limitations
The cohorts are small, historical, and not clinically validated for the users of this application. Target labels reflect each source dataset rather than a common prospective disease-risk endpoint.

## Chapter 2 — Literature and Technology Review

### 2.1 Machine learning and supervised learning
Machine learning estimates relationships from data. Supervised learning uses examples paired with labels; this project uses binary classification for the mapped source outcomes.

### 2.2 Classification and candidate algorithms
Logistic Regression provides a regularized linear baseline. Decision Trees model nonlinear rules but can have high variance. Random Forest averages randomized trees. SVM searches for a margin-based decision boundary. KNN classifies by nearby observations and depends on scaling. XGBoost builds an additive sequence of boosted trees. These are candidates, not a ranking of medical usefulness.

### 2.3 Explainable AI and SHAP
SHAP provides additive feature contributions relative to a reference sample for a model output. Here, one-hot contributions are summed back to source variables. The results explain model behavior only; they do not show causation or clinical importance.

## Chapter 3 — System Analysis

### 3.1 Existing approach
Many classroom prototypes use hardcoded predictions, show only accuracy, or omit target provenance. This project instead acquires public source data, generates metrics in code, and saves source hashes and model manifests.

### 3.2 Proposed system
A research pipeline trains versioned scikit-learn pipelines; a Flask application validates inputs and loads those pipelines for inference.

### 3.3 Functional requirements
Home/about pages; four input forms; server validation; model predictions; local explanations; model-performance and dataset pages; history; disclaimer; SQLite persistence; tests.

### 3.4 Non-functional requirements
Reproducibility, readable modules, source attribution, deterministic seed, safe SQL, no hardcoded secrets, input bounds, and explicit limitations.

### 3.5 Hardware and software
Python 3.11+, a standard laptop with enough memory for tabular modeling, a modern browser, Flask, NumPy, pandas, scikit-learn, XGBoost, SHAP, SQLite, Matplotlib, Seaborn, and Plotly.

## Chapter 4 — System Design

### 4.1 System architecture
```mermaid\nflowchart TD\nD[Dataset source]-->A[Audit and EDA]-->P[Train-fold preprocessing]-->T[CV and tuning]-->E[Held-out evaluation]-->M[Versioned model]\nU[User]-->F[Flask form]-->V[Input validation]-->I[Inference pipeline]\nM-->I\nI-->R[Result and SHAP]-->H[SQLite history]\n```

### 4.2 Data-flow diagram
```mermaid
flowchart LR
S[Public dataset sources] --> D[(Raw CSV and source manifest)]
D --> A[Validation, profiling, and EDA]
A --> T[Training pipeline and grouped CV]
T --> M[(Versioned model bundle)]
T --> R[(Metrics, tables, and figures)]
U[User] --> F[Flask form or JSON API]
F --> V[Server-side validation]
V --> M
M --> P[Prediction and SHAP service]
P --> O[Result response]
P --> H[(Local SQLite history)]
```
External data is validated and profiled before training. At inference time, user values are validated, passed through the serialized pipeline, and optionally recorded in local SQLite.

### 4.3 Use cases
```mermaid\nflowchart LR\nVisitor((Visitor))-->Choose[Choose disease module]\nVisitor-->Read[Read datasets and performance]\nChoose-->Form[Submit validated form]\nForm-->Result[Review model estimate and explanation]\nResult-->History[View or clear local history]\n```

### 4.4 Flowchart
```mermaid\nflowchart TD\nS[Start]-->D{{Select module}}\nD-->I[Enter source-defined features]\nI-->V{{Valid values and CSRF?}}\nV--No-->E[Show errors]\nV--Yes-->P[Load saved pipeline]\nP-->R[Prediction and model score]\nR-->X[Compute local SHAP if supported]\nX-->H[Store minimal history]\nH-->O[Display qualified result]\n```

### 4.5 Database design
`predictions(id, created_at, disease, input_json, prediction, probability, model_version)`. Values are parameterized; no identity/contact columns are stored. History is local and can be cleared.

### 4.6 Module design
`data.py` loads and normalizes datasets; `preprocessing.py` builds leakage-safe transforms; `training.py` runs EDA/search/evaluation; `prediction.py` loads a saved artifact; `validation.py` checks web input; `database.py` handles SQLite; Flask routes and templates render pages.

## Chapter 5 — Dataset and Preprocessing

See [Dataset sources and measured field audit](DATASET_SOURCES.md) for citations, per-feature definitions, source hashes, class counts, and missingness. The audited raw record counts are 768 diabetes, 303 heart, 583 liver, and 400 kidney. The target and feature counts are generated from downloaded files, not entered by hand.

### 5.1–5.4 Sources, features, and targets
The modules use the Pima Indians Diabetes Database via the CRAN `mlbench` package, UCI Heart Disease (Cleveland processed subset), UCI ILPD, and UCI Chronic Kidney Disease. Each target mapping is implemented in `data.py`; heart is mapped to source `num` 0 versus greater than 0, liver `Selector` 1 versus 2, kidney `ckd` versus `notckd`, and diabetes published binary outcome.

### 5.5 Missing values and data cleaning
The measured missing-value counts are shown in `DATASET_SOURCES.md`. Pima zero-coded measurements are converted to missing according to dataset documentation; legitimate zero pregnancies are preserved. Categorical values are normalized; duplicate rows are counted and kept in the dataset but grouped by feature vector during split creation.

### 5.6–5.8 Feature engineering and split
No new clinical features are invented. Numeric medians, category modes, one-hot encoding, and scaling are fitted within each training fold. Outliers are flagged by the 1.5×IQR rule but not deleted automatically. A fixed-seed grouped stratified 80/20 holdout is used; grouped 5-fold CV and search run only on training records.

## Chapter 6 — Model Development

Six candidates are evaluated per disease: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, and XGBoost. GridSearchCV uses grouped stratified five-fold CV. Selection prioritizes mean CV recall/sensitivity, with F1 then ROC-AUC as tie-breakers. The test set is not used to select a model. The saved pipeline is refit on all available rows only after holdout results are recorded.

## Chapter 7 — Implementation

### 7.1 Backend and frontend
Flask routes serve responsive Bootstrap 5 templates. Server-side schemas check required fields, types, categories, and broad input limits. A session token protects state-changing routes.

### 7.2 Database
SQLite stores a timestamp, disease key, submitted model inputs, prediction, probability estimate, and model version. Parameterized statements are used. Do not enter identifiable health information.

### 7.3 Prediction and explainability
The app loads a Joblib pipeline containing its preprocessing and estimator. Positive-class model estimates are not clinically calibrated. Permutation SHAP runs on transformed numerical features, with encoded categories grouped to the source variable where possible.

## Chapter 8 — Results

### 8.1 EDA results
The generated EDA figures include target and missingness summaries, numeric distributions, box plots, and numeric correlation heatmaps. Observations below are descriptive checks from the measured dataset profile; IQR flags identify values for review and do not establish data errors.

{eda_observations()}

### 8.2 Model results
All values below were generated from this run. Recall and precision show the sensitivity/false-alarm trade-off; scores are benchmark estimates only.

{measured_results_table(results)}

### 8.3 Candidate comparison
"""
    for key, result in results.items():
        report += f"\n#### {result['title']}\n\n{model_comparison_table(result)}\n\n"
        report += f"Selected: **{result['selected_model']}**. Confusion matrix (true rows 0/1, predicted columns 0/1): `{result['holdout_metrics']['confusion_matrix']}`. Held-out records: {result['data_quality']['test_records']}. SHAP summary status: {result['shap_summary']['status']}.\n\n"
        report += f"Figures: `reports/figures/{key}/target_and_missingness.png`, `numeric_distributions.png`, `numeric_boxplots.png`, `correlation_heatmap.png`, `model_comparison.png`, `confusion_matrix.png`, `roc_and_precision_recall.png`, `shap_summary.png`.\n\n"
    report += "### 8.4 Confusion matrices\n\n"
    report += "Each selected model's confusion matrix is reported with its candidate table above. All-candidate held-out confusion matrices are saved in `reports/figures/<module>/all_model_confusion_matrices.png`. Counts use true rows 0/1 and predicted columns 0/1.\n\n"
    report += "### 8.5 ROC and precision-recall curves\n\n"
    report += "Held-out curves for each candidate are saved in `reports/figures/<module>/all_model_roc_precision_recall.png`. Thresholds use each estimator's default decision rule and are not clinical operating points.\n\n"
    report += "### 8.6 Feature importance and SHAP analysis\n\n"
    report += "Mean absolute grouped SHAP contributions are saved as `reports/tables/<module>_shap_feature_summary.csv` and `reports/figures/<module>/shap_summary.png`. Contributions describe model behavior relative to a reference sample; they are neither causal evidence nor medical advice.\n\n"
    report += "### 8.7 Application screenshots\n\n"
    report += "The local browser review covered the home, model performance, dataset, history, about, disclaimer, and all four disease-form pages. Static browser screenshots are not included in this package; submission-specific images may be saved in `reports/screenshots/` if required.\n\n"
    report += f"""## Chapter 9 — Testing

Automated test outcome: **{test_summary}**. The suite checks schema-to-dataset agreement, input validation, model loading/inference, SQLite persistence, page responses, forms, and CSRF behavior. It does not replace an independent security assessment or external clinical validation.

{test_rows}

Manual interface review: the local browser rendered the home, performance, dataset, history, about, disclaimer, and four disease-form pages. Static browser screenshots are not included in this package; capture submission-specific images into `reports/screenshots/` if your institution requires them. This page review does not establish accessibility certification.

## Chapter 10 — Limitations and Future Scope

The Pima cohort has a narrow demographic scope. The heart, liver, and kidney datasets are small and historical. Missingness may not be random; repeated rows are handled as groups but patient identifiers are unavailable. Class prevalence, measurement practice, and target definitions differ. False positives, false negatives, bias, population shift, probability calibration, privacy, and clinical utility require further evaluation. Future work should add representative external data, independent validation, calibration and threshold review, subgroup analysis, monitoring, and formal privacy/security governance.

## Chapter 11 — Conclusion

The project implements four public-dataset binary classification modules with actual preprocessing, six candidate estimators per module, grouped cross-validation, hyperparameter search, held-out evaluation, Joblib pipelines, SHAP contribution analysis where supported, a Flask interface, and local SQLite history. The reported metrics describe only the documented benchmark splits. They do not establish medical diagnosis, prospective risk prediction, or clinical effectiveness.
"""
    (ROOT / "reports" / "PROJECT_REPORT.md").write_text(report, encoding="utf-8")
    return report


def write_viva() -> None:
    lines = ["# Viva Preparation", "", "Answers are intentionally concise; adapt them to the measured outputs in `PROJECT_REPORT.md`.", "", "## Core questions", ""]
    for index, (question, answer) in enumerate(VIVA, 1):
        lines.extend([f"### {index}. {question}", answer, ""])
    lines.extend(["## Difficult technical questions", ""])
    for index, (question, answer) in enumerate(ADVANCED_VIVA, 1):
        lines.extend([f"### D{index}. {question}", answer, ""])
    (ROOT / "reports" / "VIVA_PREPARATION.md").write_text("\n".join(lines), encoding="utf-8")


def update_readme(results: dict) -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start = "<!-- GENERATED_RESULTS_START -->"
    end = "<!-- GENERATED_RESULTS_END -->"
    section = f"\n\nGenerated from actual holdout evaluation on {date.today().isoformat()}:\n\n{measured_results_table(results)}\n\nThe metrics are single benchmark holdout estimates; see `reports/PROJECT_REPORT.md` for candidate comparisons, confusion matrices, and limitations."
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    path.write_text(left + start + section + "\n" + end + right, encoding="utf-8")


def main() -> None:
    results_path = ROOT / "reports" / "model_results" / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError("Run scripts/train_models.py before generating the report.")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    for key in DATASET_KEYS:
        load_dataset(key, ROOT / "data" / "raw")
    write_dataset_sources(results, json.loads((ROOT / "data" / "raw" / "source_manifest.json").read_text(encoding="utf-8")))
    write_report(results)
    write_viva()
    update_readme(results)
    print("Generated README results, DATASET_SOURCES.md, PROJECT_REPORT.md, and VIVA_PREPARATION.md")


if __name__ == "__main__":
    main()
