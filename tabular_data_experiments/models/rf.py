from __future__ import annotations

from typing import Any

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    CategoricalHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tabular_data_experiments.models.base_model import BaseModel


class RFModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
        n_estimators: int = 250,
        seed=42,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        if device != "cpu":
            raise ValueError("RFModel only supports CPU")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
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

        Config space as defined in
        https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/rf_config.py

        Args:
            dataset_properties: The dataset properties.

        Returns:
            The configuration space.
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                CategoricalHyperparameter("criterion", choices=["gini", "entropy"], default_value="gini"),
                CategoricalHyperparameter(
                    "max_features",
                    choices=["sqrt", "log2", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, "None"],
                    default_value="sqrt",
                ),
                CategoricalHyperparameter(
                    "max_depth", choices=["None", 2, 3, 4], default_value="None", weights=[0.7, 0.1, 0.1, 0.1]
                ),
                CategoricalHyperparameter("min_samples_split", choices=[2, 3], weights=[0.95, 0.05], default_value=2),
                UniformIntegerHyperparameter(
                    "min_samples_leaf",
                    lower=1,
                    upper=50,  # adjusted from 2 in original config to 1 to accomodate default
                    default_value=1,
                ),
                CategoricalHyperparameter("bootstrap", choices=[True, False], default_value=True),
                CategoricalHyperparameter(
                    "min_impurity_decrease",
                    choices=[0.0, 0.01, 0.02, 0.05],
                    weights=[0.85, 0.05, 0.05, 0.05],
                    default_value=0.0,
                ),
            ]
        )
        return cs

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": True}
