"""CatBoost model."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.utils import EarlyStopSet
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class CatBoostModel(BaseModel):
    def __init__(
        self,
        config: Configuration | None = None,
        dataset_properties: dict[str, Any] | None = None,
        iterations=1_000,
        od_type: Literal["Iter", "IncToDec"] = "Iter",
        early_stopping_rounds: int = 20,
        seed=42,
        device: str = "cpu",
        early_stop_set: EarlyStopSet = EarlyStopSet.TRAINING,
        auto_class_weights: Literal["Balanced", "SqrtBalanced"] | None = None,
    ) -> None:
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)
        self.early_stopping_rounds = early_stopping_rounds
        init_params = {}
        init_params["iterations"] = iterations
        init_params["od_type"] = od_type
        if self.early_stopping_rounds > 0:
            init_params["early_stopping_rounds"] = early_stopping_rounds
        init_params["task_type"] = "GPU" if device == "cuda" else "CPU"
        init_params["devices"] = "0" if device == "cuda" else None
        init_params["random_seed"] = seed
        init_params["train_dir"] = self.model_dir
        if auto_class_weights is not None:
            init_params["auto_class_weights"] = auto_class_weights
        self.early_stop_set = early_stop_set
        self.model = CatBoostClassifier(**init_params, **self.config)

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        # catboost expects categorical features to be str
        cat_features = self.dataset_properties.get("categorical_columns", None)

        X = X.infer_objects()
        if cat_features is not None:
            X = X.astype({col: "str" for col in cat_features})
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
        """
        Since catboost expects categorical features to be strings
        we do not need to encode them.
        """
        return "passthrough"

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
        **kwargs: Any,
    ) -> CatBoostModel:
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

        self.model.fit(
            X_train, y_train, cat_features=self.dataset_properties.get("categorical_columns", None), **kwargs
        )
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
        """Get the configuration space for this model as defined in
        https://github.com/kathrinse/TabSurvey/blob/de0392364f2bcb4e16ab52bbb12603e1b17c77d6/models/tree_models.py#L124.
        Default values are from documentation:
        https://catboost.ai/en/docs/references/training-parameters/common
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                UniformIntegerHyperparameter("max_depth", lower=2, upper=12, default_value=6, log=True),
                UniformFloatHyperparameter("reg_lambda", lower=0.5, upper=30, default_value=3, log=True),
                UniformFloatHyperparameter("learning_rate", lower=0.01, upper=0.3, default_value=0.03, log=True),
            ]
        )
        return cs


class CatboostEarlyStopValidModel(CatBoostModel):
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


class CatboostBalancedEarlyStopValidModel(CatboostEarlyStopValidModel):
    def __init__(
        self,
        auto_class_weights="Balanced",
        **kwargs: Any,
    ) -> None:
        kwargs.pop("auto_class_weights", None)
        super().__init__(
            auto_class_weights=auto_class_weights,
            **kwargs,
        )
