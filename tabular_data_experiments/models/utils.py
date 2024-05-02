from __future__ import annotations

from typing import Any

from enum import Enum

from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tabular_data_experiments.utils.implementations import augment_categories

PREDICT_BATCH_SIZE = 512


class EarlyStopSet(Enum):
    """Enum for early stopping."""

    VALIDATION = "validation"
    TRAINING = "training"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"early_stop_set": self.value}


def replace_str(config: dict[str, Any]) -> dict[str, Any]:
    """Replace str values representing special keywords in a
    dictionary with special keywords."""
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = replace_str(value)
        elif value == "None":
            config[key] = None
        elif value == "True":
            config[key] = True
        elif value == "False":
            config[key] = False
    return config


def get_categorical_pipeline_embedding_models() -> Pipeline:
    """Get the categorical preprocessing pipeline for embedding models."""
    categorical_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
            ),
            (
                "augment_category",
                FunctionTransformer(func=augment_categories),
            ),
        ]
    )
    return categorical_transformer
