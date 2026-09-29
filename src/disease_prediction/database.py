"""Small SQLite repository for local demonstration history."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with row access and foreign-key checking."""
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the prediction history table and indexes if absent."""
    with connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                disease TEXT NOT NULL,
                input_json TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL,
                model_version TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC)"
        )


def insert_prediction(
    database_path: str | Path,
    disease: str,
    inputs: dict[str, Any],
    prediction: int,
    probability: float | None,
    model_version: str,
) -> int:
    """Persist only the model inputs and output required for the history view."""
    created_at = datetime.now(timezone.utc).isoformat()
    input_json = json.dumps(inputs, ensure_ascii=False, allow_nan=False)
    with connect(database_path) as connection:
        cursor = connection.execute(
            """INSERT INTO predictions
               (created_at, disease, input_json, prediction, probability, model_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (created_at, disease, input_json, int(prediction), probability, model_version),
        )
        return int(cursor.lastrowid)


def list_predictions(database_path: str | Path, limit: int = 100) -> list[dict[str, Any]]:
    """Return the newest prediction records, bounded to a safe display limit."""
    safe_limit = max(1, min(int(limit), 500))
    with connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, created_at, disease, input_json, prediction, probability, model_version "
            "FROM predictions ORDER BY created_at DESC, id DESC LIMIT ?",
            (safe_limit,),
        ).fetchall()
    results = []
    for row in rows:
        record = dict(row)
        record["inputs"] = json.loads(record.pop("input_json"))
        results.append(record)
    return results


def clear_predictions(database_path: str | Path) -> None:
    """Delete local prediction history."""
    with connect(database_path) as connection:
        connection.execute("DELETE FROM predictions")
