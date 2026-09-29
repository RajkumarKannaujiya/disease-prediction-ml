from __future__ import annotations

from disease_prediction.database import clear_predictions, initialize_database, insert_prediction, list_predictions


def test_database_insert_list_and_clear(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    record_id = insert_prediction(
        database_path,
        "diabetes",
        {"Age": "34", "Glucose": "120"},
        1,
        0.73,
        "diabetes-v1.0.0",
    )
    records = list_predictions(database_path)
    assert record_id == records[0]["id"]
    assert records[0]["inputs"]["Age"] == "34"
    assert records[0]["probability"] == 0.73
    clear_predictions(database_path)
    assert list_predictions(database_path) == []


def test_database_uses_parameterized_values_and_caps_limit(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    insert_prediction(database_path, "heart", {"value": "'; DROP TABLE predictions;--"}, 0, 0.2, "heart-v1")
    records = list_predictions(database_path, limit=10000)
    assert len(records) == 1
    assert "DROP TABLE" in records[0]["inputs"]["value"]
