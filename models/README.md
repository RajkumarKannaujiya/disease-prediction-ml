# Model bundles

`python scripts/train_models.py` creates one Joblib scikit-learn pipeline per module, a metadata manifest, and a small public-data background sample used by the local SHAP demonstration. The pipeline includes imputation, encoding, scaling, and the selected estimator.

Only load Joblib files created by this project; Joblib uses pickle serialization and must not be used with untrusted artifacts. The probability shown by the application is an uncalibrated benchmark model estimate, not an individual clinical risk.
