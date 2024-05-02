"""LightGBM model."""
from __future__ import annotations

from typing import Any

import re

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from lightgbm import LGBMClassifier, early_stopping
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.utils import EarlyStopSet
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class LGBMModel(BaseModel):
    def __init__(
        self,
        config: Configuration | None = None,
        iterations: int = 1_000,
        early_stopping_rounds: int = 20,
        dataset_properties: dict[str, Any] | None = None,
        seed=42,
        device: str = "cpu",
        binary_metric: str = "auc",
        class_weight: str | None = None,
        early_stop_set: EarlyStopSet = EarlyStopSet.TRAINING,
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        if device != "cpu":
            raise ValueError("LGBMModel only supports CPU")

        self.early_stopping_rounds = early_stopping_rounds
        init_params = {}
        if self.dataset_properties.get("output_type") == "binary":
            init_params["objective"] = "binary"
            init_params["metric"] = binary_metric
        else:
            init_params["objective"] = "multiclass"
            init_params["num_class"] = self.dataset_properties.get("num_classes")
            init_params["metric"] = "multiclass"  # already means log loss

        if class_weight is not None:
            init_params["class_weight"] = class_weight
        init_params["seed"] = seed
        init_params["categorical_feature"] = self.dataset_properties.get("categorical_columns", [])
        init_params["n_estimators"] = iterations
        # add callback for early stopping
        if self.early_stopping_rounds > 0:
            self.early_stopping_callback = early_stopping(stopping_rounds=early_stopping_rounds)
        self.early_stop_set = early_stop_set
        self._internal_feature_map: dict[str, str] | None = None
        self.model = LGBMClassifier(**init_params, **self.config)

    def _prepare_data(self, X: InputDatasetType) -> InputDatasetType:

        # Avoid setting all nan column to float for lgbm
        cat_features = self.dataset_properties.get("categorical_columns", None)

        X = X.infer_objects()
        if cat_features is not None:
            X = X.astype({col: "category" for col in cat_features})

        # lightgbm does not support column names with special characters
        # See issue https://github.com/microsoft/LightGBM/issues/2455
        # Fix as based on https://github.com/autogluon/autogluon/pull/451
        if self._internal_feature_map is None:
            for column in X.columns:
                new_column = re.sub(r'[",:{}[\]]', "", column)
                if new_column != column:
                    self._internal_feature_map = {feature: i for i, feature in enumerate(list(X.columns))}
                    break

        if self._internal_feature_map is not None:
            new_columns = [self._internal_feature_map[column] for column in list(X.columns)]
            X_new = X.copy(deep=False)
            X_new.columns = new_columns
            return X_new

        return X

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
                    # ordinal encoder will propagate np.nan values for lgbm to handle
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        return categorical_transformer

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> LGBMModel:
        """Fit the model."""
        X = self._prepare_data(X)
        X_valid = self._prepare_data(X_valid) if X_valid is not None else None
        kwargs = {}
        if self.early_stopping_rounds > 0:
            X_train, X_val, y_train, y_val = self._get_early_stopping_data(
                X, y, X_val=X_valid, y_val=y_valid, early_stop_set=self.early_stop_set
            )
            # Force X train dtypes on X val as error occurs when dtypes
            # are different, happens when all nan in valid
            if hasattr(X_val, "astype"):
                X_val = X_val.astype(X_train.dtypes.to_dict())
            kwargs["eval_set"] = [(X_val, y_val)]
            kwargs["callbacks"] = [self.early_stopping_callback]
        else:
            X_train, y_train = X, y

        self.model.fit(X_train, y_train, **kwargs)
        return self

    def predict(self, X: InputDatasetType) -> np.ndarray:
        X = self._prepare_data(X)
        return super().predict(X)

    def predict_proba(self, X: InputDatasetType) -> np.ndarray:
        X = self._prepare_data(X)
        return super().predict_proba(X)

    def get_additional_run_info(self) -> dict[str, Any]:
        additional_run_info = super().get_additional_run_info()

        additional_run_info.update({"best_iteration": self.model.best_iteration_})
        return additional_run_info

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space.

        Configuration Space as defined in
        https://github.com/kathrinse/TabSurvey/blob/main/models/tree_models.py
        Default values are taken from the LightGBM documentation
        https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.LGBMClassifier.html
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter("num_leaves", lower=2, upper=4096, default_value=31, log=True),
                UniformFloatHyperparameter(
                    "lambda_l1",
                    lower=1e-8,
                    upper=10,
                    default_value=1e-8,  # Default in doc = 0 but lower bound is 1e-8
                    log=True,
                ),
                UniformFloatHyperparameter(
                    "lambda_l2",
                    lower=1e-8,
                    upper=10,
                    default_value=1e-8,  # Default in doc = 0 but lower bound is 1e-8
                    log=True,
                ),
                UniformFloatHyperparameter("learning_rate", lower=0.01, upper=0.3, default_value=0.1, log=True),
            ]
        )
        return cs


class LGBMEarlyStopValidModel(LGBMModel):
    def __init__(
        self,
        early_stop_set=EarlyStopSet.VALIDATION,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("early_stop_set", None)
        super().__init__(
            early_stop_set=early_stop_set,
            **kwargs,
        )


class LGBMLogLossModel(LGBMEarlyStopValidModel):
    def __init__(
        self,
        binary_metric="binary_logloss",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("binary_metric", None)
        super().__init__(
            binary_metric=binary_metric,
            **kwargs,
        )


class LGBMBalancedModel(LGBMEarlyStopValidModel):
    def __init__(
        self,
        class_weight="balanced",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("class_weight", None)
        super().__init__(
            class_weight=class_weight,
            **kwargs,
        )
