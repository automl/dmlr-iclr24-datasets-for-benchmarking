"""
This is the implementation of the RegCocktails as described in the paper but
with speed improvements
"""
from __future__ import annotations

from typing import Any

import os

from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
)
from ConfigSpace.configuration_space import Configuration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from tabular_data_experiments.models.cocktails.base_reg_cocktails import (
    BaseRegCocktailsModel,
)
from tabular_data_experiments.models.cocktails.utils import (
    get_updates_for_regularization_cocktails,
)


class RegCocktailsNoEmbedModel(BaseRegCocktailsModel):
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
        super().__init__(
            config=config,
            dataset_properties=dataset_properties,
            device=device,
            budget_type=budget_type,
            budget=budget,
            metric=metric,
            early_stopping=early_stopping,
            seed=seed,
            **validator_kwargs,
        )

    @staticmethod
    def get_search_space_updates(has_categorical_columns: bool = True) -> HyperparameterSearchSpaceUpdates | None:
        """Get the search space updates."""
        return get_updates_for_regularization_cocktails(
            search_embedding=False, has_categorical_columns=has_categorical_columns
        )

    def _get_encoder(self) -> tuple[str, OneHotEncoder | OrdinalEncoder]:
        """Get the encoder."""
        return ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""

        encoder = self._get_encoder()

        categorical_transformer = Pipeline(
            steps=[
                encoder,
            ]
        )
        return categorical_transformer

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
