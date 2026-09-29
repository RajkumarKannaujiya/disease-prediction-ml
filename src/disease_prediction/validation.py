"""Server-side form validation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .schemas import DISEASES


class InputValidationError(ValueError):
    """Raised when submitted values do not match the declared input schema."""

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{key}: {value}" for key, value in errors.items()))


def validate_input(disease_key: str, submitted: dict[str, Any]) -> pd.DataFrame:
    """Validate submitted values and return one-row input data in model order."""
    if disease_key not in DISEASES:
        raise InputValidationError({"disease": "Choose a supported disease module."})

    disease = DISEASES[disease_key]
    errors: dict[str, str] = {}
    values: dict[str, Any] = {}
    expected = {feature.name for feature in disease.features}
    unknown_flags = {f"{feature.name}_unknown" for feature in disease.features}

    for feature in disease.features:
        raw = submitted.get(feature.name)
        unknown_key = f"{feature.name}_unknown"
        if submitted.get(unknown_key) == "1":
            values[feature.name] = float("nan")
            continue
        if unknown_key in submitted:
            errors["form"] = "An unknown-value option was not recognized. Reload the form and try again."
            continue
        if raw is None or str(raw).strip() == "":
            errors[feature.name] = "This field is required unless you select ‘I don't know this value’."
            continue
        value = str(raw).strip()
        if feature.kind == "select":
            if value not in feature.options:
                errors[feature.name] = "Choose one of the listed values."
            else:
                values[feature.name] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[feature.name] = "Enter a valid number."
            continue
        if not math.isfinite(number):
            errors[feature.name] = "Enter a finite number."
            continue
        if feature.step == "1" and not number.is_integer():
            errors[feature.name] = "Enter a whole number."
            continue
        if feature.minimum is not None and number < feature.minimum:
            errors[feature.name] = f"Value must be at least {feature.minimum:g}."
            continue
        if feature.maximum is not None and number > feature.maximum:
            errors[feature.name] = f"Value must be no more than {feature.maximum:g}."
            continue
        values[feature.name] = int(number) if feature.step == "1" else number

    unexpected = set(submitted) - expected - unknown_flags - {"csrf_token"}
    if unexpected:
        errors["form"] = "The form contained an unexpected field. Reload the page and try again."
    if errors:
        raise InputValidationError(errors)
    return pd.DataFrame([values], columns=[feature.name for feature in disease.features])
