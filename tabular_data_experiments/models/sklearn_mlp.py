from __future__ import annotations

from typing import Any

from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from tabular_data_experiments.models.base_model import BaseModel


class SklearnMLPModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        n_iter_no_change=32,
        validation_fraction=0.1,
        tol=1e-4,
        solver="adam",
        batch_size="auto",
        shuffle=True,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        warm_start=True,
        dataset_properties: dict[str, Any] | None = None,
        seed=42,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        if device != "cpu":
            raise ValueError("MLPModel only supports CPU")

        # Reconfigures the MLP step due to the fact we can't encode tuples in ConfigSpace.

        config = self.config.copy()
        num_nodes_per_layer = config.pop("num_nodes_per_layer")
        hidden_layer_depth = config.pop("hidden_layer_depth")

        config["hidden_layer_sizes"] = tuple(num_nodes_per_layer for _ in range(hidden_layer_depth))

        self.model = MLPClassifier(
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            tol=tol,
            solver=solver,
            batch_size=batch_size,
            shuffle=shuffle,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            warm_start=warm_start,
            **config,
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

    def _get_numerical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the numerical preprocessing pipeline."""
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance_threshold", VarianceThreshold(threshold=0.0)),
                ("quantile", QuantileTransformer(output_distribution="normal")),
            ]
        )
        return numerical_transformer

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

    def get_additional_run_info(self) -> dict[str, Any]:
        additional_run_info = super().get_additional_run_info()

        additional_run_info.update({"best_iter": self.model.n_iter_})
        return additional_run_info

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """
        Get the configuration space for this model.

        Configuration Space as defined in AMLTK.
        # TODO: Add link to AMLTK

        Args:
            dataset_properties: _description_

        Returns:
            _description_
        """
        space = {
            "hidden_layer_depth": Integer("hidden_layer_depth", bounds=(1, 3), default=1),
            "num_nodes_per_layer": Integer("num_nodes_per_layer", bounds=(16, 264), log=True, default=32),
            "activation": Categorical("activation", ["tanh", "relu"], default="relu"),
            "alpha": Float("alpha", bounds=(1e-7, 1e-1), default=1e-4, log=True),
            "learning_rate_init": Float("learning_rate_init", bounds=(1e-4, 0.5), default=1e-3, log=True),
            "early_stopping": Categorical("early_stopping", [True, False], default=True),
            "learning_rate": ["constant", "invscaling", "adaptive"],
        }
        cs = ConfigurationSpace(space)
        return cs

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": True}
