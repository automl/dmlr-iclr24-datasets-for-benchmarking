""" This is the implementation of the RegCocktails as described in the paper using the code from the paper"""
from __future__ import annotations

from typing import Any

import os

from autoPyTorch.datasets.resampling_strategy import (
    HoldoutValTypes,
    NoResamplingStrategyTypes,
)
from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
)
from autoPyTorch.utils.pipeline import get_dataset_requirements
from ConfigSpace.configuration_space import Configuration

from tabular_data_experiments.models.cocktails.reg_cocktails_none import (
    RegCocktailsNoEmbedModel,
)
from tabular_data_experiments.models.cocktails.utils import (
    get_updates_for_regularization_cocktails,
)
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class RegCocktailsOldModel(RegCocktailsNoEmbedModel):
    required_keys = ["categorical_columns", "numerical_columns", "num_features"]

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
        super().__init__(
            config=config,
            dataset_properties=dataset_properties,
            device=device,
            budget_type=budget_type,
            budget=budget,
            metric=metric,
            early_stopping=early_stopping,
            seed=seed,
            validate_feature_columns=False,
        )

    @staticmethod
    def get_search_space_updates(has_categorical_columns: bool = True) -> HyperparameterSearchSpaceUpdates | None:
        """Get the search space updates."""
        return get_updates_for_regularization_cocktails(
            search_embedding=False, new_code=False, has_categorical_columns=has_categorical_columns
        )

    def _prepare_dataset(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        y_val: InputDatasetType | None = None,
        X_val: TargetDatasetType | None = None,
    ):
        """Prepare the dataset."""
        dataset_properties = self._get_dataset_properties_during_fit(X, y)
        dataset_properties["num_features"] = len(dataset_properties["feature_names"])
        dataset_info = {k: dataset_properties[k] for k in self.required_keys}

        # Initialise validator
        self.validator = self.validator.fit(
            X_train=X,
            y_train=y,
            X_test=X_val,
            y_test=y_val,
        )
        resampling_strategy = (
            NoResamplingStrategyTypes.no_resampling if X_val is None else HoldoutValTypes.stratified_holdout_validation
        )

        dataset = TabularDataset(
            X=X,
            Y=y,
            X_val=X_val,
            Y_val=y_val,
            resampling_strategy=resampling_strategy,
            validator=self.validator,
            seed=self.seed,
            # dataset_name=dataset_openml.name,
            shuffle=False,
            **dataset_info,
        )

        dataset_requirements = get_dataset_requirements(
            self.dataset_properties, search_space_updates=self.model.search_space_updates
        )
        dataset_properties = dataset.get_dataset_properties(dataset_requirements)
        return dataset, dataset_properties

    def _get_backend(self):
        from autoPyTorch.utils.backend import create

        backend = create(
            temporary_directory=os.path.join(self.model_dir, "tmp"),
            output_directory=os.path.join(self.model_dir, "out"),
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
        )

        return backend
