from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.datasets.resampling_strategy import (
    HoldoutValTypes,
    NoResamplingStrategyTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
)
from autoPyTorch.utils.pipeline import get_configuration_space, get_dataset_requirements
from ConfigSpace import Configuration, ConfigurationSpace
from numpy.typing import NDArray
from sklearn.base import check_is_fitted
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.cocktails.utils import init_fit_dictionary
from tabular_data_experiments.models.utils import PREDICT_BATCH_SIZE
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType

BINARY_THRESHOLD = 0.5


class BaseRegCocktailsModel(BaseModel):
    def __init__(
        self,
        config: Configuration | None = None,
        dataset_properties: dict[str, Any] | None = None,
        device: str = "cpu",
        budget_type: str = "epochs",
        budget: int = 105,
        metric: str = "accuracy",
        early_stopping: int = -1,
        seed=42,
        **validator_kwargs: dict[str, Any],
    ) -> None:

        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        self.budget_type = budget_type
        self.budget = budget
        self.seed = seed
        self.device = device
        self.metric = metric
        self.early_stopping = early_stopping
        has_categorical_columns = len(self.dataset_properties.get("categorical_columns", [])) > 0

        self.model: TabularClassificationPipeline = TabularClassificationPipeline(
            dataset_properties=dataset_properties,
            random_state=self.seed,
            search_space_updates=self.get_search_space_updates(has_categorical_columns=has_categorical_columns),
            config=self.configuration,
        )
        self.validator = TabularInputValidator(is_classification=True, **validator_kwargs)

    @staticmethod
    def get_search_space_updates(has_categorical_columns: bool = True) -> HyperparameterSearchSpaceUpdates | None:
        """Get the search space updates."""
        # return get_updates_for_regularization_cocktails()
        raise NotImplementedError

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        # reg cocktails again detects categorical features, so convert to category type
        cat_features = self.dataset_properties.get("categorical_columns", None)

        # set categorical columns to categorical dtype
        if cat_features is not None:
            for col in cat_features:
                if col in X.columns:
                    X[col] = X[col].astype("category")
        return X

    def preprocess_data(self, X: InputDatasetType, training: bool = True) -> TargetDatasetType:
        """Preprocess the data."""
        # Store column names to preserve order after preprocessing
        if training:
            self.preprocessor.fit(X)

        check_is_fitted(self.preprocessor)

        return self.preprocessor.transform(X)

    def get_additional_run_info(self) -> dict[str, Any]:
        additional_run_info = super().get_additional_run_info()
        num_params = self.model.named_steps["trainer"].run_summary.trainable_parameter_count
        additional_run_info["num_params"] = num_params
        return additional_run_info

    def _get_encoder(self) -> OneHotEncoder | OrdinalEncoder:
        """Get the encoder."""
        raise NotImplementedError

    def _prepare_dataset(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
    ):
        """Prepare the dataset."""
        X = self._prepare_data(X)
        self.validator = self.validator.fit(
            X_train=X,
            y_train=y,
        )
        resampling_strategy = NoResamplingStrategyTypes.no_resampling

        dataset = TabularDataset(
            X=X,
            Y=y,
            resampling_strategy=resampling_strategy,
            validator=self.validator,
            seed=self.seed,
            # dataset_name=dataset_openml.name,
            shuffle=False,
        )

        dataset_requirements = get_dataset_requirements(
            self.dataset_properties, search_space_updates=self.model.search_space_updates
        )
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        return dataset, dataset_properties

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""

        encoder = self._get_encoder()

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant")),
                encoder,
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

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> "BaseRegCocktailsModel":
        # get search space updates
        # create pipeline
        # Create a backend to store the results
        backend = self._get_backend()
        dataset, dataset_properties = self._prepare_dataset(X, y)
        backend.save_datamanager(dataset)
        fit_dictionary = init_fit_dictionary(
            dataset=dataset,
            dataset_properties=dataset_properties,
            backend=backend,
            budget_type=self.budget_type,
            budget=self.budget,
            device=self.device,
            metric_name=self.metric,
            early_stopping=self.early_stopping,
        )
        # Fit the pipeline
        self.model.fit(fit_dictionary)
        return self

    def _get_backend(self):
        raise NotImplementedError

    def predict(self, X: TargetDatasetType) -> NDArray[np.int_]:
        X, _ = self.validator.transform(X)

        predictions = self.model.predict(X, batch_size=PREDICT_BATCH_SIZE)

        threshold = BINARY_THRESHOLD
        if self.dataset_properties["output_type"] == "binary":
            predictions[predictions >= threshold] = 1
            predictions[predictions < threshold] = 0
        return predictions

    def predict_proba(self, X: TargetDatasetType) -> NDArray[np.float_]:
        X, _ = self.validator.transform(X)
        return self.model.predict_proba(X, batch_size=PREDICT_BATCH_SIZE)

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        # Requires dataset properties which must contain: numerical columns, categorical columns, task type
        # can be done using autopytorch pipeline utils
        has_categorical_columns = len(dataset_properties.get("categorical_columns", [])) > 0
        search_space_updates = cls.get_search_space_updates(has_categorical_columns=has_categorical_columns)
        return get_configuration_space(info=dataset_properties, search_space_updates=search_space_updates)

    def get_model_description(self):
        params = super().get_model_description()
        params.pop("validator", None)
        return params
