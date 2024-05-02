"""Povides functionality to run experiments on tabular data."""
from __future__ import annotations

from typing import Any

import json
import subprocess
from pathlib import Path

import pandas as pd
from ConfigSpace import Configuration

from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType

REQUIRED_ARGS = [
    "experiment_name",
    "data_loader",
    "task_id",
    "model_name",
    "metric",
    "fold_number",
    "split_number",
    "config_id",
]
SPLITS_DIR = Path(__file__).parents[2] / "splits_pq"
CONFIGS_DIR = Path(__file__).parents[2] / "configs"


def get_run_configuration_dict(config_id: int, model_name: str) -> dict[str, Any]:
    # HACK: This is a hack to use the same configs for all models
    if model_name.startswith("xgb"):
        model_name = "xgb"
    elif model_name.startswith("lgbm"):
        model_name = "lgbm"
    
    model_name = model_name.replace("_early_stop_valid", "")

    configs = json.load((CONFIGS_DIR / str(model_name) / "configs.json").open("r"))
    configuration = configs[str(config_id)]
    return configuration


def load_run_configuration(config_id: int, model_name: str, dataset_properties: dict[str, Any]) -> Configuration:
    configuration = get_run_configuration_dict(config_id, model_name)

    # HACK: This is a hack to make sure that the configuration is valid for reg cocktails
    try:
        configuration = Configuration(BaseModel.get_model_config_space(model_name, dataset_properties), configuration)
    except Exception as e:
        # happens when data does not have numerical columns.
        if model_name == "reg_cocktails" and "unknown hyperparameter imputer:numerical_strategy" in str(e):
            configuration.pop("imputer:numerical_strategy", None)
            configuration = Configuration(
                BaseModel.get_model_config_space("reg_cocktails", dataset_properties), configuration
            )
        else:
            raise e
    return configuration


def load_train_valid_test_splits(
    task_id: int, fold_number: int, split_number: int, X: InputDatasetType, y: TargetDatasetType
) -> tuple[
    InputDatasetType, TargetDatasetType, InputDatasetType, TargetDatasetType, InputDatasetType, TargetDatasetType
]:

    df = pd.read_parquet(SPLITS_DIR / str(task_id) / "splits.pq")
    train_indices = df[
        (df[f"openml_fold_{fold_number}"] != split_number) & (~df[f"openml_fold_{fold_number}"].isna())
    ].index.tolist()
    val_indices = df[df[f"openml_fold_{fold_number}"] == split_number].index.tolist()
    test_indices = df[df[f"openml_fold_{fold_number}"].isna()].index.tolist()
    print(
        f"train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}, total: {len(train_indices) + len(val_indices) + len(test_indices)}"
    )
    print(f"Total data: {X.shape[0]}")
    assert len(set(train_indices) | set(val_indices) | set(test_indices)) == X.shape[0]

    X_train, X_valid, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
    y_train, y_valid, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]
    total_length = len(set(train_indices) | set(val_indices) | set(test_indices))

    if task_id != 361681:  # skip task since splits dont include 3 samples due to stratification
        assert total_length == X.shape[0], (total_length, X.shape[0])

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_git_revision_short_hash() -> str:
    """
    code from https://stackoverflow.com/a/21901260
    """
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
