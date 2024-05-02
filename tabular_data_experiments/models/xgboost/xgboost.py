"""XGBoost model class."""
from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.utils import EarlyStopSet
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class XGBModel(BaseModel):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
        n_estimators: int = 1_000,
        early_stopping_rounds: int = 20,
        booster: str = "gbtree",
        seed=42,
        verbose: int = 2,
        device: str = "cpu",
        tree_method: str = "hist",
        enable_categorical: bool = True,
        early_stop_set: EarlyStopSet = EarlyStopSet.TRAINING,
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        if device != "cpu":
            raise ValueError("XGBModel only supports CPU")

        self.early_stopping_rounds = early_stopping_rounds

        init_params = {}
        if self.dataset_properties.get("output_type") == "binary":
            init_params["objective"] = "binary:logistic"
            init_params["eval_metric"] = "auc"
        else:
            init_params["objective"] = "multi:softprob"
            init_params["num_class"] = self.dataset_properties.get("num_classes")
            init_params["eval_metric"] = "mlogloss"

        init_params["n_estimators"] = n_estimators

        if self.early_stopping_rounds > 0:
            init_params["early_stopping_rounds"] = early_stopping_rounds

        init_params["booster"] = booster
        init_params["enable_categorical"] = (
            enable_categorical and len(self.dataset_properties.get("categorical_columns", [])) > 0
        )
        init_params["tree_method"] = tree_method  # needed for enable_categorical
        init_params["seed"] = seed
        init_params["verbosity"] = verbose
        self.early_stop_set = early_stop_set
        self.model = XGBClassifier(**init_params, **self.config)

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> XGBModel:
        """Fit the model."""
        X = self._prepare_data(X)
        X_valid = self._prepare_data(X_valid) if X_valid is not None else None

        kwargs = {}
        if self.early_stopping_rounds > 0:
            X_train, X_val, y_train, y_val = self._get_early_stopping_data(
                X, y, X_val=X_valid, y_val=y_valid, early_stop_set=self.early_stop_set
            )
            kwargs["eval_set"] = [(X_val, y_val)]
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
                    # ordinal encoder will propagate np.nan values for xgb to handle
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )
        return categorical_transformer

    def get_additional_run_info(self) -> dict[str, Any]:
        additional_run_info = super().get_additional_run_info()

        additional_run_info.update({"best_iteration": self.model.best_iteration})
        return additional_run_info

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space. Parameters are based on
        https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/xgb_config.py
        and
        https://github.com/Yura52/tabular-dl-revisiting-models/blob/main/output/adult/xgboost/tuning/0.toml.
        Default values are based on
        https://xgboost.readthedocs.io/en/stable/parameter.html
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter("max_depth", lower=1, upper=11, default_value=6),
                UniformFloatHyperparameter("gamma", lower=1e-8, upper=7, default_value=1e-8, log=True),
                UniformFloatHyperparameter("min_child_weight", lower=1, upper=100, default_value=1, log=True),
                UniformFloatHyperparameter("subsample", lower=0.5, upper=1, default_value=1),
                UniformFloatHyperparameter("colsample_bytree", lower=0.5, upper=1, default_value=1),
                UniformFloatHyperparameter("colsample_bylevel", lower=0.5, upper=1, default_value=1),
                UniformFloatHyperparameter("reg_alpha", lower=1e-8, upper=1e2, default_value=1e-8, log=True),
                UniformFloatHyperparameter("reg_lambda", lower=1e-8, upper=1e2, default_value=1e-8, log=True),
                UniformFloatHyperparameter("learning_rate", lower=1e-5, upper=0.7, default_value=0.3, log=True),
            ]
        )
        return cs


class XGBEarlyStopValidModel(XGBModel):
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


class XGBNoEarlyStopModel(XGBModel):
    def __init__(
        self,
        early_stopping_rounds: int = -1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("early_stop_set", None)
        super().__init__(
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )
