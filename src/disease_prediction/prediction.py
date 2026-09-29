"""Saved model loading and inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .explainability import local_feature_contributions


def load_model_bundle(model_root: str | Path, disease_key: str) -> tuple[Any, dict[str, Any], pd.DataFrame]:
    """Load a locally generated pipeline, metadata and small SHAP reference set."""
    model_dir = Path(model_root) / disease_key
    pipeline_path = model_dir / "pipeline.joblib"
    metadata_path = model_dir / "metadata.json"
    if not pipeline_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Trained model bundle not found for {disease_key}.")
    pipeline = joblib.load(pipeline_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    background_path = model_dir / "shap_background.csv"
    background = pd.read_csv(background_path) if background_path.exists() else pd.DataFrame()
    if not background.empty:
        categorical = set()
        for name, _, columns in pipeline.named_steps["preprocess"].transformers_:
            if name == "categorical":
                categorical.update(columns)
        for column in categorical:
            background[column] = background[column].map(
                lambda value: str(value).strip() if pd.notna(value) else float("nan")
            )
    return pipeline, metadata, background


def predict_one(
    pipeline: Any,
    metadata: dict[str, Any],
    background: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Predict a binary outcome and compute local positive-class SHAP values."""
    prediction = int(pipeline.predict(features)[0])
    probabilities = pipeline.predict_proba(features)[0]
    positive_probability = float(probabilities[1])
    contributions = local_feature_contributions(pipeline, background, features)
    return {
        "prediction": prediction,
        "positive_probability": positive_probability,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "explanations": contributions,
    }
