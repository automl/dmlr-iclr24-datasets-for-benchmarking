from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant  # noqa
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from tabular_data_experiments.models.base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
        seed=42,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        if device != "cpu":
            raise ValueError("DecisionTreeModel only supports CPU")

        self.model = DecisionTreeClassifier(
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
        """
        Get the configuration space for this model.

        Args:
            dataset_properties: The dataset properties.

        Returns:
            The configuration space.
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                Constant(
                    "criterion",
                    "gini",
                ),
                Constant(
                    "splitter",
                    "best",
                ),
                Constant(
                    "min_samples_split",
                    2,
                ),
                Constant(
                    "min_samples_leaf",
                    1,
                ),
                Constant(
                    "min_weight_fraction_leaf",
                    0.0,
                ),
            ]
        )
        return cs

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": True}
