"""Shared pytest fixtures based only on the acquired public benchmark records."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from disease_prediction.app import create_app  # noqa: E402
from disease_prediction.data import load_dataset  # noqa: E402
from disease_prediction.schemas import DISEASES  # noqa: E402


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def test_app(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-only-random-key",
            "DATABASE_PATH": str(tmp_path / "test_predictions.sqlite3"),
            "MODEL_ROOT": str(ROOT / "models"),
            "RESULTS_PATH": str(ROOT / "reports" / "model_results" / "all_results.json"),
            "QUALITY_PATH": str(ROOT / "data" / "processed" / "dataset_quality.json"),
        }
    )
    return app


@pytest.fixture
def public_row(project_root: Path):
    def make_row(disease_key: str) -> dict[str, str]:
        x, _, _ = load_dataset(disease_key, project_root / "data" / "raw")
        row = x.iloc[0].copy()
        for column in x.columns:
            if pd.isna(row[column]):
                if pd.api.types.is_numeric_dtype(x[column]):
                    row[column] = x[column].median()
                else:
                    row[column] = x[column].mode(dropna=True).iloc[0]
        values: dict[str, str] = {}
        for feature in DISEASES[disease_key].features:
            value = row[feature.name]
            if feature.kind == "select":
                # Browser select controls submit canonical option strings; the
                # public UCI heart file stores some integer-coded categories as
                # floats (for example, 0.0), so normalize those for a valid form.
                text_value = str(value)
                try:
                    numeric_value = float(text_value)
                    matching_options = [
                        option for option in feature.options
                        if float(option) == numeric_value
                    ]
                except (TypeError, ValueError):
                    matching_options = []
                values[feature.name] = (
                    matching_options[0] if matching_options else text_value
                )
            elif feature.step == "1":
                values[feature.name] = str(int(float(value)))
            else:
                values[feature.name] = str(float(value))
        return values

    return make_row


@pytest.fixture
def client_with_csrf(test_app):
    client = test_app.test_client()
    client.get("/")
    with client.session_transaction() as session:
        token = session["csrf_token"]
    return client, token
