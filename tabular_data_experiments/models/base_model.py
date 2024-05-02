"""Base Class for Models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Type

import shutil
import tempfile

import numpy as np
import tabulate
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (  # noqa
    CategoricalHyperparameter,
    Constant,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from sklearn.base import BaseEstimator, check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tabular_data_experiments.models.utils import EarlyStopSet, replace_str
from tabular_data_experiments.utils.data_utils import get_required_dataset_info
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType
from tabular_data_experiments.utils.utils import HYPERPARAMETER_NAME_TO_TABLE_NAME


class BaseModel(ABC):
    """Base Class for Models."""

    def __init__(
        self,
        seed: int = 42,
        config: Configuration | None = None,
        dataset_properties: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the model."""
        self.seed = seed
        self.dataset_properties = dataset_properties if dataset_properties is not None else {}
        self.model_dir = tempfile.mkdtemp()
        self.model: BaseEstimator = None
        if config is None:
            config = self.get_config_space(self.dataset_properties).get_default_configuration()
        self.config: dict[str, Any] = replace_str(config.get_dictionary().copy())
        self.configuration = config
        self.feature_names_in_: list[str] = self.dataset_properties.get("feature_names")
        self.init_preprocessor()

    def init_preprocessor(self) -> None:
        """Initialize the preprocessor."""
        categorical_features = self.dataset_properties.get("categorical_columns", [])
        numerical_features = self.dataset_properties.get("numerical_columns", [])
        self.preprocessor = self.get_preprocessing_pipeline(
            categorical_features=categorical_features, numerical_features=numerical_features
        )

    @classmethod
    def get_model_class(cls, model_name: str) -> Type["BaseModel"]:
        """Get the model class."""
        if model_name == "rf":
            from tabular_data_experiments.models.rf import RFModel  # noqa

            return RFModel
        elif model_name == "reg_cocktails":
            from tabular_data_experiments.models.cocktails.reg_cocktails_none import (  # noqa
                RegCocktailsNoEmbedModel,
            )

            return RegCocktailsNoEmbedModel
        elif model_name == "reg_cocktails_reduced":
            from tabular_data_experiments.models.cocktails.reg_cocktails_reduced import (  # noqa
                RegCocktailsReducedEmbeddingModel,
            )

            return RegCocktailsReducedEmbeddingModel
        elif model_name == "reg_cocktails_dim":
            from tabular_data_experiments.models.cocktails.reg_cocktails_dim import (  # noqa
                RegCocktailsDimEmbedModel,
            )

            return RegCocktailsDimEmbedModel
        elif model_name == "reg_cocktails_combined":
            from tabular_data_experiments.models.cocktails.reg_cocktails_combine import (  # noqa
                RegCocktailsCombinedEmbeddingModel,
            )

            return RegCocktailsCombinedEmbeddingModel
        elif model_name == "reg_cocktails_v1":
            from tabular_data_experiments.models.cocktails.reg_cocktails_old import (  # noqa
                RegCocktailsOldModel,
            )

            return RegCocktailsOldModel
        elif model_name == "reg_cocktails_split_onehot":
            from tabular_data_experiments.models.cocktails.reg_cocktails_split_one_hot import (  # noqa
                RegCocktailsSplitOneHotModel,
            )

            return RegCocktailsSplitOneHotModel
        elif model_name == "xgb":
            from tabular_data_experiments.models.xgboost.xgboost import XGBModel

            return XGBModel
        elif model_name == "xgb_onehot_no_early_stop":
            from tabular_data_experiments.models.xgboost.xgb_onehot import (
                XGBNoEarlyStopOneHotModel,
            )

            return XGBNoEarlyStopOneHotModel
        elif model_name == "xgb_no_early_stop":
            from tabular_data_experiments.models.xgboost.xgboost import (
                XGBNoEarlyStopModel,
            )

            return XGBNoEarlyStopModel
        elif model_name == "xgb_onehot":
            from tabular_data_experiments.models.xgboost.xgb_onehot import (
                XGBOneHotModel,
            )

            return XGBOneHotModel
        elif model_name == "xgb_early_stop_valid":
            from tabular_data_experiments.models.xgboost.xgboost import (
                XGBEarlyStopValidModel,
            )

            return XGBEarlyStopValidModel
        elif model_name == "xgb_onehot_early_stop_valid":
            from tabular_data_experiments.models.xgboost.xgb_onehot import (
                XGBOneHotEarlyStopValidModel,
            )

            return XGBOneHotEarlyStopValidModel
        elif model_name == "lgbm":
            from tabular_data_experiments.models.lightgbm import LGBMModel

            return LGBMModel
        elif model_name == "lgbm_early_stop_valid":
            from tabular_data_experiments.models.lightgbm import LGBMEarlyStopValidModel

            return LGBMEarlyStopValidModel
        elif model_name == "catboost":
            from tabular_data_experiments.models.catboost import CatBoostModel

            return CatBoostModel
        elif model_name == "catboost_early_stop_valid":
            from tabular_data_experiments.models.catboost import (
                CatboostEarlyStopValidModel,
            )

            return CatboostEarlyStopValidModel
        elif model_name == "resnet_combined":
            from tabular_data_experiments.models.resnet import ResNetModel

            ResNetModel.include_embedding_type = ["combined_embedding"]
            return ResNetModel
        elif model_name == "resnet_separate":
            from tabular_data_experiments.models.resnet import ResNetModel

            ResNetModel.include_embedding_type = ["separate_embedding"]
            return ResNetModel
        elif model_name == "resnet":
            from tabular_data_experiments.models.resnet import ResNetModel

            return ResNetModel
        elif model_name == "resnet_early_stop_valid":
            from tabular_data_experiments.models.resnet import ResNetEarlyStopValidModel

            return ResNetEarlyStopValidModel
        elif model_name == "mlp":
            from tabular_data_experiments.models.mlp import MLPModel

            return MLPModel
        elif model_name == "mlp_early_stop_valid":
            from tabular_data_experiments.models.mlp import MLPEarlyStopValidModel

            return MLPEarlyStopValidModel
        elif model_name == "mlp_combined":
            from tabular_data_experiments.models.mlp import MLPModel

            MLPModel.include_embedding_type = ["combined_embedding"]
            return MLPModel
        elif model_name == "mlp_separate":
            from tabular_data_experiments.models.mlp import MLPModel

            MLPModel.include_embedding_type = ["separate_embedding"]
            return MLPModel
        elif model_name == "ft_transformer":
            from tabular_data_experiments.models.ft_transformer import (
                FTTransformerModel,
            )

            return FTTransformerModel
        elif model_name == "ft_transformer_early_stop_valid":
            from tabular_data_experiments.models.ft_transformer import (
                FTTransformerEarlyStopValidModel,
            )

            return FTTransformerEarlyStopValidModel
        elif model_name == "saint":
            from tabular_data_experiments.models.saint.saint import SaintModel

            return SaintModel
        elif model_name == "saint_early_stop_valid":
            from tabular_data_experiments.models.saint.saint import (
                SaintEarlyStopValidModel,
            )

            return SaintEarlyStopValidModel
        elif model_name == "linear":
            from tabular_data_experiments.models.linear import LinearModel

            return LinearModel
        elif model_name == "hgbt":
            from tabular_data_experiments.models.hgbt import HGBTModel

            return HGBTModel
        elif model_name == "sklearn_mlp":
            from tabular_data_experiments.models.sklearn_mlp import SklearnMLPModel

            return SklearnMLPModel
        elif model_name == "tree":
            from tabular_data_experiments.models.tree import DecisionTreeModel

            return DecisionTreeModel
        elif model_name == "lgbm_logloss_early_stop_valid":
            from tabular_data_experiments.models.lightgbm import LGBMLogLossModel

            return LGBMLogLossModel
        elif model_name == "lgbm_balanced_early_stop_valid":
            from tabular_data_experiments.models.lightgbm import LGBMBalancedModel

            return LGBMBalancedModel
        elif model_name == "catboost_balanced_early_stop_valid":
            from tabular_data_experiments.models.catboost import (
                CatboostBalancedEarlyStopValidModel,
            )

            return CatboostBalancedEarlyStopValidModel
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    @classmethod
    def create(
        cls,
        model_name: str,
        model_kwargs: dict[str, Any],
        config: dict[str, Any] | None = None,
        dataset_properties: dict[str, Any] | None = None,
    ) -> "BaseModel":
        """Create a model."""
        model_class = cls.get_model_class(model_name)
        return model_class(config=config, dataset_properties=dataset_properties, **model_kwargs)

    def _get_categorical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the categorical preprocessing pipeline."""
        categorical_transformer = Pipeline(
            steps=[
                (
                    "ordinal",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2),
                ),
            ]
        )
        return categorical_transformer

    def _get_numerical_preprocessing_pipeline(self) -> Pipeline | str:
        """Get the numerical preprocessing pipeline."""
        numerical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ]
        )
        return numerical_transformer

    def get_preprocessing_pipeline(
        self, categorical_features: list[str], numerical_features: list[str]
    ) -> ColumnTransformer:
        """Get the preprocessing pipeline."""
        categorical_transformer = self._get_categorical_preprocessing_pipeline()
        numerical_transformer = self._get_numerical_preprocessing_pipeline()

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, categorical_features),
                ("num", numerical_transformer, numerical_features),
            ],
            verbose_feature_names_out=False,
        ).set_output(
            transform="pandas"
        )  # to reorder the columns back to original
        return preprocessor

    def preprocess_data(self, X: InputDatasetType, training: bool = True) -> TargetDatasetType:
        """Preprocess the data."""
        # Store column names to preserve order after preprocessing
        if training:
            self.preprocessor.fit(X)

        check_is_fitted(self.preprocessor)
        preprocessed_data = self.preprocessor.transform(X)  # no need to reorder columns since some may not exist.
        return preprocessed_data

    def _prepare_data(self, X: InputDatasetType) -> InputDatasetType:
        """Converts columns to expected dtypes.
        Currently, only needed for reg_cocktails, lgbm and xgb.
        Catboost implements its own preparation.
        """
        cat_features = self.dataset_properties.get("categorical_columns", None)

        X = X.infer_objects()
        if cat_features is not None:
            X = X.astype({col: "category" for col in cat_features})

        # set all nan columns as int, since we are setting the
        # enable_categorical for XGB where it is not possible to
        # have dtypes outside bool, int, float and category
        for col in X.columns:
            if X[col].isna().all():
                X[col] = X[col].astype("float")
        return X

    def _get_dataset_properties_during_fit(self, X, y):
        dataset_properties = get_required_dataset_info(
            X, y, old_categorical_columns=self.dataset_properties.get("categorical_columns", None)
        )

        return dataset_properties

    def fit(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        *,
        X_valid: InputDatasetType | None = None,
        y_valid: TargetDatasetType | None = None,
    ) -> BaseModel:
        """Fit the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: InputDatasetType) -> np.ndarray:
        """Predict using the model."""
        return self.model.predict(X)

    def predict_proba(self, X: InputDatasetType) -> np.ndarray:
        """Predict using the model."""
        return self.model.predict_proba(X)

    def score(self, X: InputDatasetType, y: TargetDatasetType) -> float:
        """Score the model."""
        return self.model.score(X, y)

    def _get_early_stopping_data(
        self,
        X: InputDatasetType,
        y: TargetDatasetType,
        train_size=0.8,
        X_val: InputDatasetType | None = None,
        y_val: TargetDatasetType | None = None,
        early_stop_set: EarlyStopSet = EarlyStopSet.VALIDATION,
    ) -> tuple[InputDatasetType, InputDatasetType, TargetDatasetType, TargetDatasetType]:
        """Split the data into train and validation sets.
        Currently, only used for LGBM, CatBoost and XGB.
        """
        if early_stop_set == EarlyStopSet.VALIDATION:
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val must be provided when using early_stop_set='valid'.")
            X_train, y_train = X, y
        elif early_stop_set == EarlyStopSet.TRAINING:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=train_size, shuffle=True, stratify=y, random_state=self.seed
            )
        else:
            raise ValueError(f"Invalid early_stop_set: {early_stop_set}")

        # Force X train dtypes on X val
        X_val = X_val.astype(X_train.dtypes.to_dict())
        return X_train, X_val, y_train, y_val

    def get_additional_run_info(self) -> dict[str, Any]:
        """Get additional run info."""
        return {}

    @classmethod
    @abstractmethod
    def get_config_space(cls, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space for the model."""
        raise NotImplementedError

    @classmethod
    def get_model_config_space(cls, model_type: str, dataset_properties: dict[str, Any]) -> ConfigurationSpace:
        """Get the configuration space for the model."""
        model_class = cls.get_model_class(model_type)
        return model_class.get_config_space(dataset_properties=dataset_properties)

    def __del__(self) -> None:
        """Delete the model."""
        shutil.rmtree(self.model_dir, ignore_errors=True)

    def get_model_description(self):
        """Returns the instance attributes as a dictionary."""
        params = self.__dict__.copy()
        params.pop("model", None)
        params.pop("model_dir", None)
        params.pop("dataset_properties", None)
        params.pop("configuration", None)
        early_stop_set = params.pop("early_stop_set", None)
        if early_stop_set is not None:
            params.update(early_stop_set.to_dict())
        preprocessor = params.pop("preprocessor", None)
        transformers_info = {}
        if preprocessor is not None:
            transformers = preprocessor.transformers
            for name, transformer, _ in transformers:
                transformers_info[name] = (
                    list(transformer.named_steps.keys()) if isinstance(transformer, Pipeline) else transformer
                )
        params["preprocessor"] = transformers_info
        return params

    def get_model_config_table(self, format: Literal["latex", "html", "markdown"] = "latex"):
        cs = self.get_config_space(self.dataset_properties)
        config_space_dict = dict(cs.items())
        first_row = ["Name", "Type", "Default", "Range"]
        rows = [first_row]
        for name, hyperparameter in config_space_dict.items():
            row = [name, HYPERPARAMETER_NAME_TO_TABLE_NAME[hyperparameter.__class__.__name__]]
            if isinstance(hyperparameter, CategoricalHyperparameter):
                row.append(hyperparameter.default_value)
                row.append(hyperparameter.choices)
            elif isinstance(hyperparameter, Constant):
                row.append(hyperparameter.value)
                row.append(hyperparameter.value)
            elif isinstance(
                hyperparameter,
                (
                    UniformFloatHyperparameter,
                    UniformIntegerHyperparameter,
                    NormalFloatHyperparameter,
                    NormalIntegerHyperparameter,
                ),
            ):
                row.append(hyperparameter.default_value)
                row.append(f"[{hyperparameter.lower}, {hyperparameter.upper}]")
            else:
                raise ValueError(f"Unknown hyperparameter type: {hyperparameter.__class__.__name__}")
            rows.append(row)
        return tabulate.tabulate(rows, headers="firstrow", tablefmt=format)

    @classmethod
    def get_properties(cls) -> dict[str, Any]:
        """Get the properties of the model."""
        return {"shortname": cls.__name__, "name": cls.__name__, "handles_sparse": False}
