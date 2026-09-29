"""Model-agnostic local SHAP contributions for submitted examples."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def local_feature_contributions(
    pipeline,
    background: pd.DataFrame,
    instance: pd.DataFrame,
    max_background: int = 12,
) -> list[dict[str, float | str]]:
    """Compute approximate local feature contributions to positive-class output.

    Contributions describe the model response, not causal or clinical effects.
    The bounded background/sample settings keep request-time work manageable.
    """
    if background.empty or instance.empty:
        return []
    try:
        reference = background.sample(n=min(max_background, len(background)), random_state=19)
        contributions = grouped_shap_values(pipeline, reference, instance)
        row = contributions.iloc[0]
        output = [{"feature": str(name), "value": float(value)} for name, value in row.items()]
        return sorted(output, key=lambda item: abs(float(item["value"])), reverse=True)[:6]
    except Exception:
        # Prediction remains available when an optional explainer backend fails.
        return []


def _to_dense(transformed) -> np.ndarray:
    return transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)


def transformed_feature_groups(pipeline, original_columns: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    """Map encoded column indexes back to the original dataset variables."""
    preprocessor = pipeline.named_steps["preprocess"]
    names = list(preprocessor.get_feature_names_out())
    groups = {column: [] for column in original_columns}
    position = 0
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        columns = list(columns)
        if transformer_name == "categorical":
            encoder = transformer.named_steps["encoder"]
            for column, categories in zip(columns, encoder.categories_, strict=False):
                count = len(categories)
                groups[column].extend(range(position, position + count))
                position += count
        else:
            for column in columns:
                groups[column].append(position)
                position += 1
    if position != len(names):
        raise ValueError("Could not map transformed model features to source variables.")
    return names, groups


def grouped_shap_values(pipeline, background: pd.DataFrame, examples: pd.DataFrame) -> pd.DataFrame:
    """Calculate permutation SHAP on numeric transformed data and group one-hot columns."""
    import shap

    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    feature_names, groups = transformed_feature_groups(pipeline, list(examples.columns))
    reference = _to_dense(preprocessor.transform(background))
    transformed_examples = _to_dense(preprocessor.transform(examples))

    def positive_probability(values):
        return estimator.predict_proba(np.asarray(values))[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.Explainer(
            positive_probability,
            reference,
            algorithm="permutation",
            feature_names=feature_names,
        )
        values = explainer(transformed_examples, max_evals=max(2 * len(feature_names) + 1, 25))
    raw_values = np.asarray(values.values)
    if raw_values.ndim == 3:
        raw_values = raw_values[:, :, -1]
    grouped = {
        column: raw_values[:, indices].sum(axis=1) if indices else np.zeros(len(examples))
        for column, indices in groups.items()
    }
    return pd.DataFrame(grouped, index=examples.index)
