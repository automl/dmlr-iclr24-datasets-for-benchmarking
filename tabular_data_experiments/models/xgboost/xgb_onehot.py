"""XGBoost model class."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tabular_data_experiments.models.utils import EarlyStopSet

from tabular_data_experiments.models.xgboost.xgboost import XGBModel
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class XGBOneHotModel(XGBModel):
    def __init__(
        self,
        tree_method: str = "auto",
        enable_categorical: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            enable_categorical=enable_categorical,
            tree_method=tree_method,
            **kwargs,
        )

    def get_preprocessing_pipeline(
        self, categorical_features: list[str], numerical_features: list[str]
    ) -> ColumnTransformer:
        """Get the preprocessing pipeline."""
        # sklearn when no numerical features are present raises an error
        # For a sparse output, all columns should be a numeric or convertible to a numeric.
        numerical_transformer = self._get_numerical_preprocessing_pipeline()
        categorical_transformer = self._get_categorical_preprocessing_pipeline()
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            verbose_feature_names_out=False,
        )
        return preprocessor

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

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> XGBModel:
        """Fit the model."""
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
        return super(XGBModel, self).predict(X)

    def predict_proba(self, X: InputDatasetType) -> np.ndarray:
        return super(XGBModel, self).predict_proba(X)


class XGBNoEarlyStopOneHotModel(XGBOneHotModel):
    def __init__(
        self,
        early_stopping_rounds: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            early_stopping_rounds=early_stopping_rounds,
            **kwargs,
        )


class XGBOneHotEarlyStopValidModel(XGBOneHotModel):
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
