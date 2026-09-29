from __future__ import annotations

from disease_prediction.data import load_dataset
from disease_prediction.prediction import load_model_bundle, predict_one
from disease_prediction.validation import validate_input


def test_model_loading_and_prediction_function(project_root, public_row):
    pipeline, metadata, background = load_model_bundle(project_root / "models", "diabetes")
    features = validate_input("diabetes", public_row("diabetes"))
    result = predict_one(pipeline, metadata, background, features)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["positive_probability"] <= 1.0
    assert result["model_version"] == metadata["model_version"]
    assert result["explanations"]


def test_every_saved_bundle_loads_and_predicts(project_root):
    for key in ("diabetes", "heart", "liver", "kidney"):
        pipeline, metadata, _ = load_model_bundle(project_root / "models", key)
        x, _, _ = load_dataset(key, project_root / "data" / "raw")
        raw_record = x.iloc[[0]].copy()
        for column in raw_record.columns:
            if raw_record[column].isna().any():
                if str(raw_record[column].dtype).startswith("string") or raw_record[column].dtype == object:
                    raw_record[column] = raw_record[column].fillna(x[column].mode(dropna=True).iloc[0])
                else:
                    raw_record[column] = raw_record[column].fillna(x[column].median())
        prediction = pipeline.predict(raw_record)[0]
        assert int(prediction) in (0, 1)
        assert metadata["feature_names"] == list(x.columns)
