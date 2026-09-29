"""Leakage-safe sklearn preprocessing pipelines."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build transformers from training-fold column dtypes only."""
    numeric = list(features.select_dtypes(include=["number"]).columns)
    categorical = [column for column in features.columns if column not in numeric]
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ]
    categorical_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline(categorical_steps), categorical))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline(features: pd.DataFrame, estimator: Any) -> Pipeline:
    """Pair preprocessing and estimator so inference repeats training transforms."""
    return Pipeline([("preprocess", make_preprocessor(features)), ("model", estimator)])
