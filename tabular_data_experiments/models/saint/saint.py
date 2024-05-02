"""Model for Saint"""
from __future__ import annotations

from typing import Any

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    CategoricalHyperparameter,
    Constant,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.model_library.TabSurvey.models.saint import SAINT
from tabular_data_experiments.models.saint.utils import get_saint_config
from tabular_data_experiments.models.utils import (
    EarlyStopSet,
    get_categorical_pipeline_embedding_models,
)
from tabular_data_experiments.utils.data_utils import get_required_dataset_info
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class SaintModel(BaseModel):
    """Model for FTTransformer"""

    def __init__(
        self,
        config: Configuration | None = None,
        dataset_properties: dict[str, Any] | None = None,
        device: str = "cpu",
        early_stopping_rounds: int = 16,
        epochs: int = 100,
        seed=42,
        early_stop_set: EarlyStopSet = EarlyStopSet.TRAINING,
    ) -> None:
        """
        Initialise the ResNet model.

        Args:
            config: _description_. Defaults to None.
            dataset_properties: _description_. Defaults to None.
            device: _description_. Defaults to "cpu".
            use_checkpoints: _description_. Defaults to False.
            checkpoint_dir: _description_. Defaults to None.
            batch_size: _description_. Defaults to 128.
            es_patience: _description_. Defaults to 40.
            lr_patience: _description_. Defaults to 30.
            verbose: _description_. Defaults to 0.
            seed: _description_. Defaults to 42.
        """
        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        self.seed = seed
        # TODO: Fix reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = device

        if self.device == "cuda":
            cudnn.deterministic = True
            cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            torch.cuda.init()
        self.device = device
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.early_stop_set = early_stop_set

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

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""
        return get_categorical_pipeline_embedding_models()

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> "SaintModel":
        """Fit the model."""

        # Since, imputation and ordinal encoding is done the dataset properties
        # are different from the original dataset
        dataset_properties = get_required_dataset_info(
            X, y, old_categorical_columns=self.dataset_properties.get("categorical_columns", None)
        )

        categorical_indicator = dataset_properties.get("categorical_indicator", None)
        num_features = X.shape[1]
        # get cardinality of categorical features from
        # original dataset and add +1 for unknown
        cat_dims_without_na = self.dataset_properties.get("n_categories_per_cat_col", None)
        cat_dims = [dim + 1 for dim in cat_dims_without_na]
        n_classes = dataset_properties.get("n_classes", 1)
        n_classes = n_classes if n_classes > 2 else 1
        objective = "classification" if n_classes > 2 else "binary"
        args, params = get_saint_config(
            self.config,
            categorical_indicator=categorical_indicator,
            cat_dims=cat_dims,
            epochs=self.epochs,
            device=self.device,
            output_dir=Path(self.model_dir),
            early_stopping_rounds=self.early_stopping_rounds,
            num_features=num_features,
            objective=objective,
            num_classes=n_classes,
        )
        self.model = SAINT(args=args, params=params)
        # split for early stopping
        if self.early_stopping_rounds > 0:
            X_train, X_val, y_train, y_val = self._get_early_stopping_data(
                X, y, X_val=X_valid, y_val=y_valid, early_stop_set=self.early_stop_set
            )
            X_val = X_val.to_numpy().astype("float32")
            y_val = y_val.to_numpy().astype("float32")
        else:
            X_train, X_val, y_train, y_val = X, None, y, None

        X_train = X_train.to_numpy().astype("float32")
        y_train = y_train.to_numpy().astype("float32")
        # assumes X, y are numpy arrays
        self.model.fit(X_train, y_train, X_val, y_val)
        return self

    def get_additional_run_info(self) -> dict[str, Any]:
        """Currently only returns the number of parameters"""
        additional_info = super().get_additional_run_info()
        additional_info.update({"num_params": self.model.get_model_size()})
        # best_epoch
        additional_info.update({"best_iteration": self.model.min_val_loss_epoch})
        return additional_info

    def predict(self, X: InputDatasetType) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        return super().predict(X)

    def predict_proba(self, X: InputDatasetType) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        return super().predict_proba(X)

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """
        Configuration Space as defined in
        https://github.com/kathrinse/TabSurvey/blob/main/models/saint.py
        Default values taken from the original SAINT paper
        https://arxiv.org/abs/2106.01342
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                CategoricalHyperparameter(
                    "args__batch_size", [32, 64, 256], default_value=64
                ),  # based on paper settings
                Constant("args__lr", 0.0001),
                Constant("args__val_batch_size", 32),
                CategoricalHyperparameter("params__depth", [1, 2, 3, 6, 12], default_value=6),
                CategoricalHyperparameter("params__heads", [2, 4, 8], default_value=8),
                CategoricalHyperparameter("params__dim", [32, 64, 128, 256], default_value=32),
                CategoricalHyperparameter(
                    "params__dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], default_value=0.1
                ),
            ]
        )
        return cs


class SaintEarlyStopValidModel(SaintModel):
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
