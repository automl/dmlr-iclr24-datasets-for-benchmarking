from __future__ import annotations

from typing import Any

import math

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
)
from sklearn.base import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class LinearModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
        seed=42,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)
        if device != "cpu":
            raise ValueError("LinearModel only supports CPU")

        self.model = LogisticRegression(
            **self.config,
            random_state=seed,
        )

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""
        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=int),
                ),
            ]
        )
        return categorical_transformer

    def get_preprocessing_pipeline(
        self, categorical_features: list[str], numerical_features: list[str]
    ) -> ColumnTransformer:
        """Get the preprocessing pipeline."""
        categorical_transformer = self._get_categorical_preprocessing_pipeline()
        numerical_transformer = self._get_numerical_preprocessing_pipeline()
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_features),
                ("num", numerical_transformer, numerical_features),
            ],
            verbose_feature_names_out=False,
        )
        return preprocessor

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space for this model.
        Configuration Space as discussed in MM channel.
        """
        cs = ConfigurationSpace()

        cs.add_hyperparameters(
            [
                CategoricalHyperparameter("penalty", ["l2", "None"], default_value="l2"),
                CategoricalHyperparameter("fit_intercept", [True, False], default_value=True),
                UniformFloatHyperparameter("C", lower=1e-12, upper=math.log(5.0), default_value=1.0, log=True),
            ]
        )
        return cs

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": True}
