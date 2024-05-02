"""Model for ResNet"""
from __future__ import annotations

from typing import Any

import random

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
from skorch import NeuralNet

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.model_library.skorch_models import (
    create_skorch_model,
)
from tabular_data_experiments.models.utils import (
    EarlyStopSet,
    get_categorical_pipeline_embedding_models,
)
from tabular_data_experiments.utils.data_utils import get_required_dataset_info
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType
from tabular_data_experiments.utils.utils import count_parameters


class FTTransformerModel(BaseModel):
    """Model for FTTransformer"""

    def __init__(
        self,
        config: Configuration | None = None,
        dataset_properties: dict[str, Any] | None = None,
        device: str = "cpu",
        use_checkpoints: bool = False,
        batch_size: int = 64,
        es_patience: int = 16,
        lr_patience: int = 30,
        verbose: int = 0,
        max_epochs: int = 100,
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

        self.device = device
        self.use_checkpoints = use_checkpoints
        self.batch_size = batch_size
        self.early_stopping_rounds = es_patience
        self.lr_patience = lr_patience
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.early_stop_set = early_stop_set

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""
        return get_categorical_pipeline_embedding_models()

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

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> "FTTransformerModel":
        """Fit the model."""

        # Since, imputation and ordinal encoding is done the dataset properties
        # are different from the original dataset
        dataset_properties = get_required_dataset_info(
            X, y, old_categorical_columns=self.dataset_properties.get("categorical_columns", None)
        )

        categorical_indicator = dataset_properties.get("categorical_indicator", None)
        num_numerical_features = len(dataset_properties.get("numerical_columns", []))
        # get cardinality of categorical features from
        # original dataset and add +1 for unknown
        cat_dims_without_na = self.dataset_properties.get("n_categories_per_cat_col", None)
        cat_dims = [dim + 1 for dim in cat_dims_without_na]
        # As done in LeoGrin/tabular-benchmark: https://www.github.com/LeoGrin/tabular-benchmark
        self.config["module__d_token"] = (self.config.pop("d_token") // self.config["module__n_heads"]) * self.config[
            "module__n_heads"
        ]
        self.model: NeuralNet = create_skorch_model(
            config=self.config,
            model_type="transformer",
            use_checkpoints=self.use_checkpoints,
            checkpoint_dir=self.model_dir,
            es_patience=self.early_stopping_rounds,
            lr_patience=self.lr_patience,
            verbose=self.verbose,
            device=self.device,
            categorical_indicator=categorical_indicator,
            categories=cat_dims,
            output_shape=self.dataset_properties.get("output_shape", 1),
            num_numerical_features=num_numerical_features,
            n_classes=self.dataset_properties.get("n_classes", 2),
            max_epochs=self.max_epochs,
        )
        # Converting dataframe to numpy array since skorch throws error
        # Error:
        #   batch must contain tensors, numpy arrays, numbers, dicts or lists;
        #   found <class 'pandas.core.arrays.categorical.Categorical'>
        if self.early_stopping_rounds > 0:
            X_train, X_val, y_train, y_val = self._get_early_stopping_data(
                X, y, X_val=X_valid, y_val=y_valid, early_stop_set=self.early_stop_set
            )
        else:
            X_train, y_train = X, y
        self.model.fit(
            X_train.to_numpy().astype("float32"),
            y_train,
            X_val=X_val.to_numpy().astype("float32") if X_val is not None else None,
            y_val=y_val if y_val is not None else None,
        )
        return self

    def predict(self, X: InputDatasetType) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        return super().predict(X)

    def predict_proba(self, X: InputDatasetType) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy().astype(np.float32)
        return super().predict_proba(X)

    def get_additional_run_info(self) -> dict[str, Any]:
        additional_run_info = super().get_additional_run_info()
        num_params = count_parameters(self.model.module_)
        additional_run_info.update({"num_params": num_params})
        # get best iteration from skorch
        early_stopping_callback = {k: v for (k, v) in self.model.callbacks_}["EarlyStopping"]
        additional_run_info.update(
            {
                "best_iteration": early_stopping_callback.best_epoch_,
            }
        )
        return additional_run_info

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """
        Get the configuration space for this model.
        Configuration as defined in
        https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/ft_transformer_config.py

        Args:
            dataset_properties: Dataset properties.

        Returns:
            ConfigurationSpace
        """
        cs = ConfigurationSpace()
        cs.add_hyperparameters(
            [
                Constant("module__activation", "reglu"),
                Constant("module__token_bias", "True"),
                Constant("module__prenormalization", "True"),
                CategoricalHyperparameter("batch_size", [64, 256, 512, 1024], default_value=256),
                CategoricalHyperparameter("module__kv_compression", [True, False], default_value=True),
                CategoricalHyperparameter(
                    "module__kv_compression_sharing", ["headwise", "key-value"], default_value="headwise"
                ),
                Constant("module__initialization", "kaiming"),
                UniformIntegerHyperparameter("module__n_layers", 1, 6, default_value=3, q=1),
                Constant("module__n_heads", 8),
                UniformFloatHyperparameter("module__d_ffn_factor", 2.0 / 3, 8.0 / 3, default_value=4.0 / 3),
                UniformFloatHyperparameter("module__ffn_dropout", 0, 0.5, default_value=0.1),
                UniformFloatHyperparameter("module__attention_dropout", 0.0, 0.5, default_value=0.2),
                UniformFloatHyperparameter("module__residual_dropout", 0.0, 0.5, default_value=0),
                UniformFloatHyperparameter("lr", 1e-5, 1e-3, log=True, default_value=1e-4),
                UniformFloatHyperparameter("optimizer__weight_decay", 1e-8, 1e-3, log=True, default_value=1e-5),
                UniformIntegerHyperparameter("d_token", 64, 512, default_value=192, q=1),
                CategoricalHyperparameter("lr_scheduler", [True, False], default_value=False),
                Constant("optimizer", "adamw"),
            ]
        )
        return cs


class FTTransformerEarlyStopValidModel(FTTransformerModel):
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
