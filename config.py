"""Environment-driven configuration for local and production Flask runs."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("DISEASE_APP_SECRET_KEY") or secrets.token_hex(32)
    DATABASE_PATH = os.environ.get("DISEASE_APP_DATABASE", str(BASE_DIR / "database" / "predictions.db"))
    MODEL_ROOT = os.environ.get("DISEASE_APP_MODEL_ROOT", str(BASE_DIR / "models"))
    RESULTS_PATH = os.environ.get("DISEASE_APP_RESULTS", str(BASE_DIR / "reports" / "model_results" / "all_results.json"))
    QUALITY_PATH = os.environ.get("DISEASE_APP_QUALITY", str(BASE_DIR / "data" / "processed" / "dataset_quality.json"))
    # Prediction inputs can contain health data, so storing them is opt-in.
    STORE_PREDICTION_HISTORY = os.environ.get("DISEASE_APP_STORE_HISTORY", "0") == "1"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SECURE = os.environ.get("DISEASE_APP_COOKIE_SECURE", "0") == "1"
    SESSION_COOKIE_SAMESITE = "Lax"
    MAX_CONTENT_LENGTH = 64 * 1024
