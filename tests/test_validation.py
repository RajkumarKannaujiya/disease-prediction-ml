from __future__ import annotations

import pytest

from disease_prediction.schemas import DISEASES, get_feature_columns
from disease_prediction.validation import InputValidationError, validate_input
from disease_prediction.data import load_dataset


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_actual_dataset_columns_match_form_schema(project_root, disease_key):
    x, _, _ = load_dataset(disease_key, project_root / "data" / "raw")
    assert list(x.columns) == get_feature_columns(disease_key)


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_valid_public_record_values_pass_schema(project_root, public_row, disease_key):
    result = validate_input(disease_key, public_row(disease_key))
    assert result.shape == (1, len(DISEASES[disease_key].features))


def test_missing_required_input_is_rejected(public_row):
    values = public_row("diabetes")
    values.pop("Glucose")
    with pytest.raises(InputValidationError, match="required"):
        validate_input("diabetes", values)


def test_non_numeric_value_is_rejected(public_row):
    values = public_row("diabetes")
    values["Age"] = "twenty"
    with pytest.raises(InputValidationError, match="valid number"):
        validate_input("diabetes", values)


def test_out_of_range_and_unknown_field_are_rejected(public_row):
    values = public_row("heart")
    values["age"] = "999"
    values["invented_feature"] = "1"
    with pytest.raises(InputValidationError) as error:
        validate_input("heart", values)
    assert "age" in error.value.errors
    assert "form" in error.value.errors


def test_unsupported_module_is_rejected():
    with pytest.raises(InputValidationError):
        validate_input("unknown", {})
