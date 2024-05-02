"""Experimental"""
from __future__ import annotations

from typing import Any

import os

import pandas as pd
from autoPyTorch.data.tabular_validator import TabularInputValidator
from autoPyTorch.pipeline.tabular_classification import TabularClassificationPipeline
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
)
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import (  # noqa
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.base import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.models.cocktails.base_reg_cocktails import (
    BaseRegCocktailsModel,
)
from tabular_data_experiments.models.cocktails.utils import (
    get_updates_for_regularization_cocktails,
)
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class RegCocktailsSplitOneHotModel(BaseRegCocktailsModel):
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
    ) -> None:

        super().__init__(seed=seed, config=config, dataset_properties=dataset_properties)

        self.budget_type = budget_type
        self.budget = budget
        self.seed = seed
        self.device = device
        self.metric = metric
        self.early_stopping = early_stopping
        has_categorical_columns = len(self.dataset_properties.get("categorical_columns", [])) > 0

        pipeline_config = {k: v for k, v in self.config.items() if k != "min_categories_for_embedding"}
        model = TabularClassificationPipeline(
            dataset_properties=dataset_properties,
            random_state=self.seed,
            search_space_updates=self.get_search_space_updates(has_categorical_columns=has_categorical_columns),
        )
        configuration_space = model.get_hyperparameter_search_space()
        pipeline_configuration = Configuration(configuration_space, pipeline_config)
        self.model: TabularClassificationPipeline = model.set_hyperparameters(pipeline_configuration)
        self.validator = TabularInputValidator(is_classification=True)

    @staticmethod
    def get_encode_embed_split(
        categorical_features: list[Any], n_categories_per_cat_col: list[int], min_categories_for_embedding: int = 3
    ) -> tuple[list[Any], list[Any]]:
        embed = [
            feat
            for feat, card in zip(categorical_features, n_categories_per_cat_col)
            if card > min_categories_for_embedding
        ]
        encode = [
            feat
            for feat, card in zip(categorical_features, n_categories_per_cat_col)
            if card <= min_categories_for_embedding
        ]

        return encode, embed

    def init_preprocessor(self) -> None:
        """Initialize the preprocessor."""
        categorical_features = self.dataset_properties.get("categorical_columns", [])
        n_categories_per_cat_col = self.dataset_properties.get("n_categories_per_cat_col", [])
        encode_features, embed_features = self.get_encode_embed_split(
            categorical_features, n_categories_per_cat_col, self.config["min_categories_for_embedding"]
        )

        numerical_features = self.dataset_properties.get("numerical_columns", [])
        numerical_transformer = self._get_numerical_preprocessing_pipeline()
        onehotencoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        ordinalencoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.embed_features = embed_features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("encode", onehotencoder, encode_features),
                ("embed", ordinalencoder, embed_features),
            ],
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

    def _prepare_data(self, X: pd.DataFrame) -> pd.DataFrame:
        # reg cocktails again detects categorical features, so convert to category type

        if self.embed_features is not None:
            for col in self.embed_features:
                if col in X.columns:
                    X[col] = X[col].astype("category")
        return X

    def preprocess_data(self, X: InputDatasetType, training: bool = True) -> TargetDatasetType:
        """Preprocess the data."""
        # Store column names to preserve order after preprocessing
        if training:
            self.preprocessor.fit(X)

        check_is_fitted(self.preprocessor)
        transformed_X = self.preprocessor.transform(X)

        # set categorical columns to categorical dtype
        return self._prepare_data(transformed_X)

    @staticmethod
    def get_search_space_updates(has_categorical_columns: bool = True) -> HyperparameterSearchSpaceUpdates | None:
        """Get the search space updates."""
        return get_updates_for_regularization_cocktails(
            search_embedding=False, embedding_choice="combined", has_categorical_columns=has_categorical_columns
        )

    def _get_backend(self):
        from autoPyTorch.automl_common.common.utils.backend import create

        backend = create(
            prefix="cocktails",
            temporary_directory=os.path.join(self.model_dir, "tmp"),
            output_directory=os.path.join(self.model_dir, "out"),
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
        )

        return backend

    @classmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space."""
        cs = super().get_config_space(dataset_properties)
        cs.add_hyperparameter(
            UniformIntegerHyperparameter("min_categories_for_embedding", lower=3, upper=7, default_value=3)
        )
        return cs
