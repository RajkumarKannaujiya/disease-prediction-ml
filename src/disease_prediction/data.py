"""Dataset loading, target normalization and data-quality summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schemas import get_feature_columns


DATASET_KEYS = ("diabetes", "heart", "liver", "kidney")
PIMA_ZERO_IS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def _clean_name(name: Any) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    aliases = {
        "outcome": "Outcome",
        "pregnant": "Pregnancies",
        "pressure": "BloodPressure",
        "triceps": "SkinThickness",
        "mass": "BMI",
        "pedigree": "DiabetesPedigreeFunction",
        "diabetes": "Outcome",
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "bloodpressure": "BloodPressure",
        "blood_pressure": "BloodPressure",
        "skinthickness": "SkinThickness",
        "skin_thickness": "SkinThickness",
        "insulin": "Insulin",
        "bmi": "BMI",
        "diabetespedigreefunction": "DiabetesPedigreeFunction",
        "diabetes_pedigree_function": "DiabetesPedigreeFunction",
        "age": "age",
        "gender": "gender",
        "sex": "sex",
        "cp": "cp",
        "trestbps": "trestbps",
        "chol": "chol",
        "fbs": "fbs",
        "restecg": "restecg",
        "thalach": "thalach",
        "exang": "exang",
        "oldpeak": "oldpeak",
        "slope": "slope",
        "ca": "ca",
        "thal": "thal",
        "num": "num",
        "target": "target",
        "total_bilirubin": "total_bilirubin",
        "totalbilirubin": "total_bilirubin",
        "tb": "total_bilirubin",
        "direct_bilirubin": "direct_bilirubin",
        "directbilirubin": "direct_bilirubin",
        "db": "direct_bilirubin",
        "alkphos": "alkphos",
        "alkaline_phosphotase": "alkphos",
        "alkaline_phosphatase": "alkphos",
        "sgpt": "sgpt",
        "alamine_aminotransferase": "sgpt",
        "alanine_aminotransferase": "sgpt",
        "sgot": "sgot",
        "aspartate_aminotransferase": "sgot",
        "total_protiens": "total_proteins",
        "total_proteins": "total_proteins",
        "tp": "total_proteins",
        "alb": "albumin",
        "albumin": "albumin",
        "a_g_ratio": "ag_ratio",
        "ag_ratio": "ag_ratio",
        "selector": "Selector",
        "sg": "sg",
        "al": "al",
        "su": "su",
        "rbc": "rbc",
        "pc": "pc",
        "pcc": "pcc",
        "ba": "ba",
        "bgr": "bgr",
        "bu": "bu",
        "sc": "sc",
        "sod": "sod",
        "pot": "pot",
        "hemo": "hemo",
        "pcv": "pcv",
        "wc": "wc",
        "wbcc": "wc",
        "rc": "rc",
        "rbcc": "rc",
        "htn": "htn",
        "dm": "dm",
        "cad": "cad",
        "appet": "appet",
        "pe": "pe",
        "ane": "ane",
        "class": "class",
    }
    return aliases.get(value, value)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [_clean_name(column) for column in result.columns]
    if result.columns.duplicated().any():
        raise ValueError(f"Column normalization produced duplicate names: {list(result.columns)}")
    return result


def _target_column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str:
    for choice in choices:
        if choice in frame.columns:
            return choice
    raise ValueError(f"Could not locate target column. Found: {list(frame.columns)}")


def load_dataset(key: str, raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load and normalize one downloaded dataset without using test information."""
    if key not in DATASET_KEYS:
        raise KeyError(f"Unknown dataset: {key}")
    path = raw_dir / f"{key}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run scripts/download_data.py first.")
    frame = _normalize_columns(pd.read_csv(path, na_values=["?", "NA", "N/A", ""], keep_default_na=True))
    if key == "diabetes" and "age" in frame.columns:
        frame = frame.rename(columns={"age": "Age"})
    if key == "diabetes":
        target_name = "Outcome"
    elif key == "heart":
        target_name = _target_column(frame, ("num", "target"))
    elif key == "liver":
        target_name = _target_column(frame, ("Selector", "target"))
    else:
        target_name = _target_column(frame, ("class", "target"))

    expected = get_feature_columns(key)
    missing_features = sorted(set(expected) - set(frame.columns))
    if missing_features:
        raise ValueError(f"{key}: expected features missing from downloaded data: {missing_features}")
    x = frame[expected].copy()
    raw_y = frame[target_name]

    if key == "heart":
        numeric_target = pd.to_numeric(raw_y, errors="coerce")
        y = (numeric_target > 0).astype("int8")
        y[numeric_target.isna()] = np.nan
    elif key == "liver":
        selector = pd.to_numeric(raw_y, errors="coerce")
        y = (selector == 1).astype("int8")
        y[selector.isna()] = np.nan
        # Verify published class counts to catch target reversal or source drift.
        observed = y.dropna().value_counts().to_dict()
        if observed.get(1, 0) != 416 or observed.get(0, 0) != 167:
            raise ValueError(f"ILPD target mapping did not match published counts: {observed}")
    elif key == "kidney":
        normalized_target = raw_y.astype("string").str.strip().str.lower()
        y = normalized_target.map({"ckd": 1, "notckd": 0, "not ckd": 0})
    else:
        numeric_target = pd.to_numeric(raw_y, errors="coerce")
        if numeric_target.notna().any():
            y = numeric_target
        else:
            y = raw_y.astype("string").str.strip().str.lower().map(
                {"pos": 1, "positive": 1, "tested_positive": 1, "neg": 0, "negative": 0, "tested_negative": 0}
            )

    # Standardize nominal values while preserving their meaning. Numeric-coded
    # nominal variables stay categorical so a numeric distance is not implied.
    categorical_by_dataset = {
        "heart": {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"},
        "liver": {"gender"},
        "kidney": {"sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"},
        "diabetes": set(),
    }
    categorical_columns = categorical_by_dataset[key]
    for column in expected:
        if column in categorical_columns:
            values = x[column].astype("string").str.strip().str.lower()
            if key == "kidney" and column == "sg":
                numeric_values = pd.to_numeric(values, errors="coerce")
                values = numeric_values.map(lambda item: f"{item:.3f}" if pd.notna(item) else pd.NA).astype("string")
            elif key == "kidney" and column in {"al", "su"}:
                numeric_values = pd.to_numeric(values, errors="coerce")
                values = numeric_values.map(lambda item: str(int(item)) if pd.notna(item) and float(item).is_integer() else pd.NA).astype("string")
            x[column] = values.astype(object).where(values.notna(), np.nan)
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    if key == "liver":
        x["gender"] = x["gender"].str.capitalize()

    # In Pima, zero is a documented missing-value code for measurements where
    # zero is physiologically implausible. Pregnancies=0 remains a valid value.
    if key == "diabetes":
        for column in PIMA_ZERO_IS_MISSING:
            x.loc[pd.to_numeric(x[column], errors="coerce") == 0, column] = np.nan

    valid_target = pd.to_numeric(y, errors="coerce").notna()
    x = x.loc[valid_target].reset_index(drop=True)
    y = pd.to_numeric(y.loc[valid_target], errors="coerce").astype("int8").reset_index(drop=True)
    duplicates = int(pd.concat([x, y.rename("target")], axis=1).duplicated().sum())
    quality = {
        "dataset_key": key,
        "source_records": int(len(frame)),
        "records_after_target_cleaning": int(valid_target.sum()),
        "exact_duplicate_rows_detected": duplicates,
        "exact_duplicates_removed": 0,
        "modeling_records": int(len(x)),
        "features": expected,
        "feature_types": {column: str(dtype) for column, dtype in x.dtypes.items()},
        "missing_by_feature": {column: int(count) for column, count in x.isna().sum().items()},
        "target_counts": {str(label): int(count) for label, count in y.value_counts().sort_index().items()},
        "outlier_iqr_counts": {},
    }
    for column in x.select_dtypes(include=["number"]).columns:
        values = x[column].dropna()
        if not values.empty:
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            quality["outlier_iqr_counts"][column] = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
    return x, y, quality


def load_all(raw_dir: Path) -> tuple[dict[str, tuple[pd.DataFrame, pd.Series]], dict[str, dict[str, Any]]]:
    datasets: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    quality: dict[str, dict[str, Any]] = {}
    for key in DATASET_KEYS:
        x, y, report = load_dataset(key, raw_dir)
        datasets[key] = (x, y)
        quality[key] = report
    return datasets, quality


def write_quality_reports(quality: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    rows = []
    for key, report in quality.items():
        for feature in report["features"]:
            rows.append({"dataset": key, "feature": feature, "missing_count": report["missing_by_feature"][feature], "dtype": report["feature_types"][feature]})
    pd.DataFrame(rows).to_csv(output_dir / "missing_values.csv", index=False)
