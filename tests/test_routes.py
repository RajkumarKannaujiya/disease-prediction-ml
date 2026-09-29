from __future__ import annotations

import json

import pytest

from disease_prediction.schemas import DISEASES


@pytest.mark.parametrize(
    "path",
    ["/", "/about", "/prediction", "/performance", "/datasets", "/history", "/disclaimer"],
)
def test_page_routes_return_success(test_app, path):
    response = test_app.test_client().get(path)
    assert response.status_code == 200


def test_favicon_is_served(test_app):
    response = test_app.test_client().get("/static/favicon.svg")
    assert response.status_code == 200
    assert b"<svg" in response.data


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_each_disease_form_loads(test_app, disease_key):
    response = test_app.test_client().get(f"/predict/{disease_key}")
    assert response.status_code == 200
    assert DISEASES[disease_key].title.encode() in response.data


def test_invalid_form_submission_returns_validation_message(client_with_csrf):
    client, csrf_token = client_with_csrf
    response = client.post(
        "/predict/diabetes", data={"csrf_token": csrf_token, "Age": "34"}
    )
    assert response.status_code == 400
    assert b"This field is required" in response.data


def test_post_without_csrf_is_rejected(test_app):
    response = test_app.test_client().post("/history/clear", data={})
    assert response.status_code == 400


def test_valid_prediction_route_records_history(test_app, public_row):
    client = test_app.test_client()
    client.get("/")
    with client.session_transaction() as session:
        csrf_token = session["csrf_token"]
    response = client.post(
        "/predict/diabetes",
        data={"csrf_token": csrf_token, **public_row("diabetes")},
    )
    assert response.status_code == 200
    assert b"Prediction result" in response.data
    history = client.get("/history")
    assert b"diabetes-v1.0.0" in history.data


def test_json_route_rejects_invalid_payload(client_with_csrf):
    client, csrf_token = client_with_csrf
    response = client.post(
        "/api/predict/diabetes",
        data=json.dumps({"Age": "34"}),
        content_type="application/json",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 400
    assert b"errors" in response.data
