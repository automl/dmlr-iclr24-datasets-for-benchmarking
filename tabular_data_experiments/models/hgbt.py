from __future__ import annotations

from typing import Any

import sys

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
)
from pandas import value_counts
from sklearn.base import check_is_fitted
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType

MAX_BINS = 255


class HGBTModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.2,
        n_iter_no_change: int = 20,
        seed=42,
        device: str = "cpu",
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)
        self.infrequent_columns: list[str] | None = None
        if device != "cpu":
            raise ValueError("HGBTModel only supports CPU")

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            max_bins=MAX_BINS,
            **self.config,
            random_state=seed,
            categorical_features=self.dataset_properties.get("categorical_columns", None),
        )

    def _get_numerical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the numerical preprocessing pipeline."""
        numerical_transformer = Pipeline(
            steps=[
                ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ]
        )
        return numerical_transformer

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""
        categorical_transformer = Pipeline(
            steps=[
                (
                    "ordinal",
                    # ordinal encoder will propagate np.nan values for hgbt to handle
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        return categorical_transformer

    def preprocess_data(self, X: InputDatasetType, training: bool = True) -> TargetDatasetType:
        """Preprocess the data."""
        # Store column names to preserve order after preprocessing
        if training:

            # check for high cardinality categorical columns
            # since hgbt does not support categorical columns
            # with cardinality > 255
            self._check_for_infrequent_categories(X)
            # map infrequent categories to a single category
            if self.infrequent_columns is not None:
                X = self._replace_infrequent_categories(X)
            self.preprocessor.fit(X)

        check_is_fitted(self.preprocessor)
        # map infrequent categories to a single category
        if self.infrequent_columns is not None and not training:
            X = self._replace_infrequent_categories(X)
        return self.preprocessor.transform(X)

    def _check_for_infrequent_categories(self, X):
        """
        Check for infrequent categories in categorical columns.
        Leaving the top MAX_BINS - 1 categories alone and collects
        the rest for each column.

        Args:
            X: _description_
        """
        n_cat_per_col = self.dataset_properties["n_categories_per_cat_col"]
        categorical_columns = self.dataset_properties["categorical_columns"]
        high_cardinality_cols = [col for col, n_cat in zip(categorical_columns, n_cat_per_col) if n_cat > MAX_BINS]
        if len(high_cardinality_cols) > 0:
            self.infrequent_columns = {}
            for col in high_cardinality_cols:
                value_counts_ = value_counts(X[col]).sort_values(ascending=False)
                # select the rest of the categories
                infrequent = value_counts_.index.tolist()[MAX_BINS - 1 :]
                self.infrequent_columns[col] = infrequent

    def _replace_infrequent_categories(self, X):
        for col, infrequent in self.infrequent_columns.items():
            X[col] = X[col].replace(infrequent, "infrequent")
        return X

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space.

        Configuration Space as defined in
        https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/hgbt_config.py
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                NormalFloatHyperparameter(
                    "learning_rate",
                    mu=float(np.log(0.01)),
                    sigma=float(np.log(10.0)),
                    lower=sys.float_info.min,
                    upper=2**16 - 1,
                    log=True,
                ),
                NormalIntegerHyperparameter("max_leaf_nodes", mu=31, sigma=5, lower=2, upper=2**16 - 1),
                CategoricalHyperparameter("max_depth", choices=["None", 2, 3, 4], weights=[0.1, 0.1, 0.7, 0.1]),
                NormalIntegerHyperparameter(
                    "min_samples_leaf",
                    mu=20,
                    sigma=2,
                    lower=1,  # 1 is the minimum value for min_samples_leaf
                    upper=2**16 - 1,
                ),
            ]
        )
        return cs

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": True}
