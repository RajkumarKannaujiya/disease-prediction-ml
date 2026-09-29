"""Flask application factory for the local academic demonstration."""

from __future__ import annotations

import hmac
import json
import os
import secrets
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, abort, current_app, flash, jsonify, redirect, render_template, request, session, url_for
from config import Config

from .database import clear_predictions, initialize_database, insert_prediction, list_predictions
from .prediction import load_model_bundle, predict_one
from .schemas import DISEASES, schema_to_dict
from .validation import InputValidationError, validate_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def _csrf_valid(value: str | None) -> bool:
    expected = session.get("csrf_token", "")
    return bool(value and expected and hmac.compare_digest(expected, value))


def create_app(test_config: dict | None = None) -> Flask:
    """Create the configured Flask application."""
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)
        if test_config.get("TESTING") and "STORE_PREDICTION_HISTORY" not in test_config:
            app.config["STORE_PREDICTION_HISTORY"] = True
    initialize_database(app.config["DATABASE_PATH"])

    @app.context_processor
    def inject_shared_template_data():
        return {
            "csrf_token": _csrf_token(),
            "diseases": DISEASES,
            "history_enabled": app.config["STORE_PREDICTION_HISTORY"],
        }

    @app.get("/")
    def index():
        return render_template("index.html", title="Home")

    @app.get("/about")
    def about():
        return render_template("about.html", title="About the project")

    @app.get("/prediction")
    def prediction_index():
        return render_template("prediction.html", title="Choose a module")

    @app.route("/predict/<disease_key>", methods=["GET", "POST"])
    def predict(disease_key: str):
        disease = DISEASES.get(disease_key)
        if disease is None:
            abort(404)
        errors: dict[str, str] = {}
        submitted = request.form.to_dict() if request.method == "POST" else {}
        if request.method == "POST":
            if not _csrf_valid(request.form.get("csrf_token")):
                abort(400, description="The form session expired. Reload the page and try again.")
            try:
                features = validate_input(disease_key, submitted)
                pipeline, metadata, background = load_model_bundle(app.config["MODEL_ROOT"], disease_key)
                result = predict_one(pipeline, metadata, background, features)
                inputs: dict[str, str | None] = {}
                display_inputs: dict[str, str] = {}
                feature_labels = {feature.name: feature.label for feature in disease.features}
                for key, value in features.iloc[0].to_dict().items():
                    if pd.isna(value):
                        inputs[key] = None
                        display_inputs[feature_labels[key]] = "Not provided; imputed by the model"
                    else:
                        text_value = str(value)
                        inputs[key] = text_value
                        display_inputs[feature_labels[key]] = text_value
                for item in result.get("explanations", []):
                    item["feature"] = feature_labels.get(item["feature"], item["feature"])
                if app.config["STORE_PREDICTION_HISTORY"]:
                    insert_prediction(
                        app.config["DATABASE_PATH"], disease_key, inputs,
                        result["prediction"], result["positive_probability"], result["model_version"],
                    )
                return render_template(
                    "result.html", title="Prediction result", disease=disease,
                    result=result, input_values=display_inputs,
                    class_label=(disease.class_1_label if result["prediction"] == 1 else disease.class_0_label),
                )
            except InputValidationError as exc:
                errors = exc.errors
                return render_template(
                    "disease_form.html", title=f"{disease.title} estimate", disease=disease,
                    errors=errors, submitted=submitted,
                ), 400
            except FileNotFoundError:
                flash("This model has not been trained yet. Run the documented training command first.", "warning")
                return redirect(url_for("predict", disease_key=disease_key))
            except Exception as exc:  # Guard user-facing route; detailed values stay in local logs.
                current_app.logger.exception("Prediction failed for module %s", disease_key)
                flash("The prediction could not be completed. Check the local application log.", "danger")
                return redirect(url_for("predict", disease_key=disease_key))
        return render_template(
            "disease_form.html", title=f"{disease.title} estimate", disease=disease,
            errors=errors, submitted=submitted,
        )

    @app.post("/api/predict/<disease_key>")
    def api_predict(disease_key: str):
        if disease_key not in DISEASES:
            abort(404)
        if not _csrf_valid(request.headers.get("X-CSRF-Token")):
            abort(400, description="A valid session CSRF token is required.")
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Send a JSON object containing every model input."}), 400
        try:
            features = validate_input(disease_key, payload)
            pipeline, metadata, background = load_model_bundle(app.config["MODEL_ROOT"], disease_key)
            result = predict_one(pipeline, metadata, background, features)
            result["disease"] = disease_key
            result["class_label"] = DISEASES[disease_key].class_1_label if result["prediction"] == 1 else DISEASES[disease_key].class_0_label
            result["disclaimer"] = "Research estimate only; this dataset classification is not a medical diagnosis or decision tool."
            if app.config["STORE_PREDICTION_HISTORY"]:
                inputs = {
                    key: (None if pd.isna(value) else str(value))
                    for key, value in features.iloc[0].to_dict().items()
                }
                insert_prediction(database_path=app.config["DATABASE_PATH"], disease=disease_key, inputs=inputs, prediction=result["prediction"], probability=result["positive_probability"], model_version=result["model_version"])
            return jsonify(result)
        except InputValidationError as exc:
            return jsonify({"errors": exc.errors}), 400
        except FileNotFoundError:
            return jsonify({"error": "Model bundle is not available. Train the models first."}), 503

    @app.get("/history")
    def history():
        enabled = app.config["STORE_PREDICTION_HISTORY"]
        records = list_predictions(app.config["DATABASE_PATH"]) if enabled else []
        return render_template("history.html", title="Prediction history", records=records, history_enabled=enabled)

    @app.post("/history/clear")
    def history_clear():
        if not app.config["STORE_PREDICTION_HISTORY"]:
            abort(404)
        if not _csrf_valid(request.form.get("csrf_token")):
            abort(400, description="The form session expired. Reload the page and try again.")
        clear_predictions(app.config["DATABASE_PATH"])
        flash("Local prediction history cleared.", "success")
        return redirect(url_for("history"))

    @app.get("/performance")
    def performance():
        path = Path(app.config["RESULTS_PATH"])
        results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return render_template("performance.html", title="Model performance", results=results)

    @app.get("/datasets")
    def datasets():
        path = Path(app.config["QUALITY_PATH"])
        quality = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return render_template("datasets.html", title="Dataset notes", quality=quality, disease_info={key: schema_to_dict(value) for key, value in DISEASES.items()})

    @app.get("/disclaimer")
    def disclaimer():
        return render_template("disclaimer.html", title="Disclaimer")

    @app.errorhandler(404)
    def not_found(error):
        return render_template("error.html", title="Page not found", message="The requested page could not be found."), 404

    @app.errorhandler(400)
    def bad_request(error):
        return render_template("error.html", title="Request could not be processed", message=getattr(error, "description", "Invalid request.")), 400

    return app
