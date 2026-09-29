"""Reproducible EDA, model search, evaluation and artifact generation."""

from __future__ import annotations

import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .data import DATASET_KEYS, load_dataset, write_quality_reports
from .explainability import grouped_shap_values
from .preprocessing import make_pipeline
from .schemas import DISEASES


SEED = 42
SCORING = {
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _candidate_models(y_train: pd.Series) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    positive = max(int((y_train == 1).sum()), 1)
    negative = max(int((y_train == 0).sum()), 1)
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"model__max_depth": [3, 6, None], "model__min_samples_leaf": [2, 5]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample", random_state=SEED, n_jobs=1
            ),
            {"model__n_estimators": [150], "model__max_depth": [None, 8], "model__min_samples_leaf": [1, 3]},
        ),
        "Support Vector Machine": (
            CalibratedClassifierCV(
                estimator=SVC(class_weight="balanced", probability=False, random_state=SEED),
                method="sigmoid",
                cv=3,
            ),
            {"model__estimator__C": [0.5, 2.0], "model__estimator__kernel": ["rbf"]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(n_jobs=1),
            {"model__n_neighbors": [5, 11], "model__weights": ["uniform", "distance"]},
        ),
        "XGBoost": (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=SEED,
                n_jobs=1,
                scale_pos_weight=negative / positive,
            ),
            {"model__n_estimators": [100], "model__max_depth": [2, 3], "model__learning_rate": [0.05, 0.1]},
        ),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    return str(value)


def _save_eda(key: str, x: pd.DataFrame, y: pd.Series, output_dir: Path) -> dict[str, Any]:
    figure_dir = output_dir / "figures" / key
    figure_dir.mkdir(parents=True, exist_ok=True)
    combined = x.copy()
    combined["target"] = y.to_numpy()
    summary: dict[str, Any] = {
        "shape": [int(x.shape[0]), int(x.shape[1])],
        "dtypes": {column: str(dtype) for column, dtype in x.dtypes.items()},
        "describe": x.describe(include="all").replace({np.nan: None}).to_dict(),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "missing_counts": {str(k): int(v) for k, v in x.isna().sum().items()},
        "class_imbalance_ratio": float(y.value_counts().max() / max(y.value_counts().min(), 1)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=y, ax=axes[0], color="#3b82f6")
    axes[0].set_title(f"{DISEASES[key].title}: target distribution")
    axes[0].set_xlabel("Dataset target class (0/1)")
    axes[0].set_ylabel("Record count")
    missing = x.isna().sum().sort_values(ascending=False)
    if missing.max() > 0:
        sns.barplot(x=missing.index, y=missing.values, ax=axes[1], color="#e76f51")
        axes[1].tick_params(axis="x", rotation=70)
    else:
        axes[1].text(0.5, 0.5, "No missing values in selected feature columns", ha="center", va="center")
        axes[1].set_xticks([])
    axes[1].set_title("Missing values by feature")
    axes[1].set_xlabel("Feature")
    axes[1].set_ylabel("Missing record count")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_and_missingness.png", dpi=150)
    plt.close(fig)

    numeric = x.select_dtypes(include=["number"])
    if not numeric.empty:
        ncols = 3
        nrows = int(np.ceil(len(numeric.columns) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows))
        for axis, feature in zip(np.asarray(axes).ravel(), numeric.columns, strict=False):
            sns.histplot(data=combined, x=feature, hue="target", element="step", stat="density", common_norm=False, ax=axis)
            axis.set_title(f"{feature} distribution by target")
            axis.set_xlabel(feature)
            axis.set_ylabel("Density")
        for axis in np.asarray(axes).ravel()[len(numeric.columns):]:
            axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(figure_dir / "numeric_distributions.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.4 * nrows))
        for axis, feature in zip(np.asarray(axes).ravel(), numeric.columns, strict=False):
            sns.boxplot(data=combined, x="target", y=feature, ax=axis, color="#90caf9")
            axis.set_title(f"{feature} by target")
            axis.set_xlabel("Dataset target class (0/1)")
            axis.set_ylabel(feature)
        for axis in np.asarray(axes).ravel()[len(numeric.columns):]:
            axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(figure_dir / "numeric_boxplots.png", dpi=150)
        plt.close(fig)

        correlation = numeric.assign(target=y.to_numpy()).corr(numeric_only=True)
        fig, axis = plt.subplots(figsize=(max(8, len(correlation) * 0.75), max(6, len(correlation) * 0.65)))
        sns.heatmap(correlation, cmap="vlag", center=0, annot=len(correlation) <= 10, fmt=".2f", ax=axis)
        axis.set_title(f"{DISEASES[key].title}: numeric-feature correlation")
        fig.tight_layout()
        fig.savefig(figure_dir / "correlation_heatmap.png", dpi=150)
        plt.close(fig)

    categorical = [column for column in x.columns if column not in numeric.columns]
    for feature in categorical:
        table = pd.crosstab(x[feature].fillna("Missing"), y, normalize="index")
        axis = table.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="Set2")
        axis.set_title(f"{feature} distribution by target class")
        axis.set_xlabel(feature)
        axis.set_ylabel("Proportion within feature category")
        axis.legend(title="Dataset target class")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figure_dir / f"categorical_{feature}.png", dpi=150)
        plt.close()
    return summary


def _evaluate(pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall_sensitivity": recall_score(y_test, predictions, zero_division=0),
        "specificity": float(matrix[0, 0] / max(matrix[0, 0] + matrix[0, 1], 1)),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "brier_score": float(np.mean((probabilities - y_test.to_numpy()) ** 2)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    metrics = {key: (float(value) if isinstance(value, (np.floating, float)) else value) for key, value in metrics.items()}
    return metrics, predictions, probabilities


def _save_evaluation_plots(key: str, model_name: str, y_test: pd.Series, predictions: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    figure_dir = output_dir / "figures" / key
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Class 0", "Class 1"], cmap="Blues", ax=axis, colorbar=False)
    axis.set_title(f"{DISEASES[key].title}: {model_name} confusion matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def _save_candidate_diagnostics(key: str, estimators: dict[str, Any], x_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    """Save a confusion matrix and ROC/PR curves for every tuned candidate."""
    from sklearn.metrics import ConfusionMatrixDisplay

    figure_dir = output_dir / "figures" / key
    names = list(estimators)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows))
    for axis, name in zip(np.asarray(axes).ravel(), names, strict=False):
        ConfusionMatrixDisplay.from_estimator(
            estimators[name], x_test, y_test, labels=[0, 1],
            display_labels=["Class 0", "Class 1"], cmap="Blues", ax=axis, colorbar=False,
        )
        axis.set_title(f"{name}: held-out confusion matrix")
    for axis in np.asarray(axes).ravel()[len(names):]:
        axis.set_visible(False)
    fig.suptitle(f"{DISEASES[key].title}: all candidate confusion matrices")
    fig.tight_layout()
    fig.savefig(figure_dir / "all_model_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, estimator in estimators.items():
        probabilities = estimator.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        precision, recall, _ = precision_recall_curve(y_test, probabilities)
        axes[0].plot(fpr, tpr, label=name)
        axes[1].plot(recall, precision, label=name)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance reference")
    axes[0].set_title("ROC curves by candidate")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate / sensitivity")
    axes[1].set_title("Precision-recall curves by candidate")
    axes[1].set_xlabel("Recall / sensitivity")
    axes[1].set_ylabel("Precision")
    for axis in axes:
        axis.legend(fontsize="small")
    fig.suptitle(f"{DISEASES[key].title}: held-out candidate curves")
    fig.tight_layout()
    fig.savefig(figure_dir / "all_model_roc_precision_recall.png", dpi=160)
    plt.close(fig)


def _save_shap_summary(key: str, pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Generate a bounded SHAP summary over held-out records when supported."""
    try:
        reference = x_train.sample(n=min(20, len(x_train)), random_state=SEED)
        explained = x_test.sample(n=min(24, len(x_test)), random_state=SEED)
        contribution = grouped_shap_values(pipeline, reference, explained)
        mean_abs = contribution.abs().mean(axis=0)
        summary = pd.DataFrame({"feature": mean_abs.index, "mean_absolute_shap": mean_abs.to_numpy()}).sort_values("mean_absolute_shap", ascending=False)
        summary.to_csv(output_dir / "tables" / f"{key}_shap_feature_summary.csv", index=False)
        figure_dir = output_dir / "figures" / key
        figure_dir.mkdir(parents=True, exist_ok=True)
        display = summary.sort_values("mean_absolute_shap", ascending=True).tail(12)
        fig, axis = plt.subplots(figsize=(8.5, max(4, 0.35 * len(display))))
        axis.barh(display["feature"], display["mean_absolute_shap"], color="#176c71")
        axis.set_title(f"{DISEASES[key].title}: mean absolute SHAP contribution")
        axis.set_xlabel("Mean absolute contribution to class 1 model estimate")
        axis.set_ylabel("Source feature")
        fig.tight_layout()
        fig.savefig(figure_dir / "shap_summary.png", dpi=160)
        plt.close(fig)
        return {"status": "generated", "explained_holdout_rows": int(len(explained)), "mean_absolute_contributions": summary.to_dict(orient="records")}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(fpr, tpr, label=model_name, color="#2a9d8f")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance reference")
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate / sensitivity")
    axes[0].legend()
    axes[1].plot(recall, precision, label=model_name, color="#e76f51")
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall / sensitivity")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    fig.suptitle(f"{DISEASES[key].title}: held-out test set")
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_and_precision_recall.png", dpi=160)
    plt.close(fig)


def train_one(
    key: str,
    raw_dir: Path,
    model_root: Path,
    results_dir: Path,
    figure_root: Path,
    source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Tune candidates on training folds and evaluate once on a held-out split."""
    x, y, quality = load_dataset(key, raw_dir)
    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError(f"{key}: insufficient class examples for stratified evaluation.")
    eda_summary = _save_eda(key, x, y, figure_root)
    duplicate_groups = pd.util.hash_pandas_object(x.astype("string"), index=False).astype(str)
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_indices, test_indices = next(outer_cv.split(x, y, groups=duplicate_groups))
    x_train, x_test = x.iloc[train_indices].copy(), x.iloc[test_indices].copy()
    y_train, y_test = y.iloc[train_indices].copy(), y.iloc[test_indices].copy()
    groups_train = duplicate_groups.iloc[train_indices]
    quality["train_records"] = int(len(x_train))
    quality["test_records"] = int(len(x_test))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    comparison: list[dict[str, Any]] = []
    best_estimators: dict[str, Any] = {}

    for model_name, (estimator, parameter_grid) in _candidate_models(y_train).items():
        pipeline = make_pipeline(x_train, estimator)
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring=SCORING,
            refit="recall",
            cv=cv,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(x_train, y_train, groups=groups_train)
        best_estimators[model_name] = search.best_estimator_
        metrics, _, _ = _evaluate(search.best_estimator_, x_test, y_test)
        row = {
            "model": model_name,
            "cv_recall_sensitivity": float(search.best_score_),
            "cv_precision": float(search.cv_results_["mean_test_precision"][search.best_index_]),
            "cv_f1": float(search.cv_results_["mean_test_f1"][search.best_index_]),
            "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][search.best_index_]),
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall_sensitivity": metrics["recall_sensitivity"],
            "test_specificity": metrics["specificity"],
            "test_f1": metrics["f1"],
            "test_roc_auc": metrics["roc_auc"],
            "test_average_precision": metrics["average_precision"],
            "test_brier_score": metrics["brier_score"],
            "test_confusion_matrix": metrics["confusion_matrix"],
            "best_parameters": search.best_params_,
        }
        comparison.append(row)
        print(f"{key}: {model_name} CV recall={row['cv_recall_sensitivity']:.3f}, test recall={row['test_recall_sensitivity']:.3f}")

    comparison.sort(key=lambda row: (row["cv_recall_sensitivity"], row["cv_f1"], row["cv_roc_auc"]), reverse=True)
    _save_candidate_diagnostics(key, best_estimators, x_test, y_test, figure_root)
    selected = comparison[0]["model"]
    selected_pipeline = best_estimators[selected]
    final_metrics, predictions, probabilities = _evaluate(selected_pipeline, x_test, y_test)
    _save_evaluation_plots(key, selected, y_test, predictions, probabilities, figure_root)
    shap_summary = _save_shap_summary(key, selected_pipeline, x_train, x_test, figure_root)

    # Model-comparison chart contains measured CV and test metrics only.
    comparison_frame = pd.DataFrame(comparison)
    chart_cols = ["cv_recall_sensitivity", "test_recall_sensitivity", "test_precision", "test_f1", "test_roc_auc"]
    axis = comparison_frame.set_index("model")[chart_cols].plot(kind="bar", figsize=(12, 6), ylim=(0, 1), color=["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"])
    axis.set_title(f"{DISEASES[key].title}: measured model comparison")
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.tick_params(axis="x", rotation=25)
    axis.figure.tight_layout()
    axis.figure.savefig(figure_root / "figures" / key / "model_comparison.png", dpi=160)
    plt.close(axis.figure)

    model_dir = model_root / key
    model_dir.mkdir(parents=True, exist_ok=True)
    # The final saved pipeline is fitted on all rows after its holdout score was
    # recorded from the untouched split; that score remains a holdout estimate.
    selected_pipeline.fit(x, y)
    artifact_path = model_dir / "pipeline.joblib"
    joblib.dump(selected_pipeline, artifact_path)
    background = x.sample(n=min(20, len(x)), random_state=SEED)
    background.to_csv(model_dir / "shap_background.csv", index=False)
    metadata = {
        "disease_key": key,
        "disease_title": DISEASES[key].title,
        "model_name": selected,
        "model_version": f"{key}-v1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "feature_names": list(x.columns),
        "positive_class": 1,
        "target_description": DISEASES[key].target_description,
        "population_limit": DISEASES[key].population_limit,
        "dataset_source": DISEASES[key].dataset_source,
        "dataset_license": DISEASES[key].dataset_license,
        "source_sha256": (source_hashes or {}).get(key),
        "records_after_cleaning": len(x),
        "train_records": len(x_train),
        "test_records": len(x_test),
        "holdout_metrics": final_metrics,
        "shap_summary": shap_summary,
        "selection_rule": "Highest 5-fold grouped training cross-validation sensitivity/recall, ties broken by F1 then ROC-AUC; exact feature duplicates stayed in one fold; test split was not used for selection.",
        "probability_note": "Model probability estimate; not clinically calibrated or a personal medical risk.",
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    comparison_path = results_dir / f"{key}_model_comparison.csv"
    comparison_frame.drop(columns=["best_parameters"]).to_csv(comparison_path, index=False)
    comparison_json = results_dir / f"{key}_model_comparison.json"
    comparison_json.write_text(json.dumps(comparison, indent=2, default=_json_default), encoding="utf-8")
    result = {
        "disease_key": key,
        "title": DISEASES[key].title,
        "selected_model": selected,
        "selection_rule": metadata["selection_rule"],
        "target_counts": quality["target_counts"],
        "data_quality": quality,
        "eda": eda_summary,
        "holdout_metrics": final_metrics,
        "shap_summary": shap_summary,
        "model_comparison": comparison,
        "model_version": metadata["model_version"],
    }
    (results_dir / f"{key}_results.json").write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


def train_all(root: Path) -> dict[str, Any]:
    """Train all modules and write dataset-wide reports."""
    raw_dir = root / "data" / "raw"
    model_root = root / "models"
    results_dir = root / "reports" / "model_results"
    figure_root = root / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "source_manifest.json"
    source_hashes = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hashes = {item["dataset_key"]: item["sha256"] for item in manifest}
    all_results = {}
    all_quality = {}
    for key in DATASET_KEYS:
        result = train_one(key, raw_dir, model_root, results_dir, figure_root, source_hashes)
        all_results[key] = result
        all_quality[key] = result["data_quality"]
    static_images = root / "static" / "images"
    static_images.mkdir(parents=True, exist_ok=True)
    for key in DATASET_KEYS:
        figure_dir = figure_root / "figures" / key
        for source_name, destination_name in (
            ("model_comparison.png", f"{key}_model_comparison.png"),
            ("roc_and_precision_recall.png", f"{key}_roc_and_precision_recall.png"),
            ("all_model_confusion_matrices.png", f"{key}_all_model_confusion_matrices.png"),
            ("all_model_roc_precision_recall.png", f"{key}_all_model_roc_precision_recall.png"),
            ("shap_summary.png", f"{key}_shap_summary.png"),
        ):
            source = figure_dir / source_name
            if source.exists():
                shutil.copy2(source, static_images / destination_name)
    write_quality_reports(all_quality, root / "data" / "processed")
    (results_dir / "all_results.json").write_text(json.dumps(all_results, indent=2, default=_json_default), encoding="utf-8")
    rows = []
    for key, result in all_results.items():
        metrics = result["holdout_metrics"]
        rows.append({"disease": key, "selected_model": result["selected_model"], **{name: metrics[name] for name in ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc", "average_precision", "brier_score")}})
    pd.DataFrame(rows).to_csv(root / "reports" / "tables" / "selected_model_metrics.csv", index=False)
    return all_results
