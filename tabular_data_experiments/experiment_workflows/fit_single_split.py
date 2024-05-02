"""Povides functionality to run experiments on tabular data."""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import pandas as pd

from tabular_data_experiments.data_loaders.base_loader import BaseLoader
from tabular_data_experiments.experiment_workflows.utils import (
    get_git_revision_short_hash,
    load_run_configuration,
    load_train_valid_test_splits,
)
from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.run_result_config import RunConfig, RunResult
from tabular_data_experiments.target_function.target_function import TargetFunction
from tabular_data_experiments.target_function.utils import (
    StatusType,
    TargetFunctionInfo,
    TargetFunctionResult,
)
from tabular_data_experiments.utils.data_utils import get_required_dataset_info


def run_single_split_experiment(
    run_config: RunConfig,
    exp_dir: Path,
) -> RunResult:
    """
    Runs an experiment on a dataset.

    Args:
        run_config: Configuration for the experiment.
        exp_dir: Path to the experiment directory.
    """
    # Create experiment folder
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Pass openml task id to environment variable
    os.environ["OPENML_TASK_ID"] = str(run_config.task_id)

    print(f"Running experiment {run_config.experiment_name} in {exp_dir}.")
    # Create experiment log

    # Log experiment name
    # Load data
    time.time()
    data_loader = BaseLoader.create(run_config.data_loader, loader_kwargs={"task_id": run_config.task_id})
    # Here we have X, y, categorical_indicator, attribute_names
    X, y, categorical_indicator, attribute_names = data_loader.get_data()
    model_properties = BaseModel.get_model_class(model_name=run_config.model_name).get_properties()

    if X.dtypes.apply(pd.api.types.is_sparse).any() and not model_properties["handles_sparse"]:
        X = X.sparse.to_dense()

    loader_additional_info = {"attribute_names": attribute_names}
    # Store attribute names to result dictionary and dataset attributes
    # Preprocess data

    dataset_properties = get_required_dataset_info(X, y, categorical_indicator)

    # Split train to train valid
    # get splits from splits dir
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_train_valid_test_splits(
        task_id=run_config.task_id,
        fold_number=run_config.fold_number,
        split_number=run_config.split_id,
        X=X,
        y=y,
    )

    # get config from configs dir
    configuration = load_run_configuration(
        config_id=run_config.config_id, model_name=run_config.model_name, dataset_properties=dataset_properties
    )
    # Split train to train valid
    # Train best model on train data

    target_function = TargetFunction(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        model_name=run_config.model_name,
        dataset_properties=dataset_properties,
        model_kwargs=run_config.model_kwargs,
        metric=run_config.metric,
        additional_metrics=run_config.additional_metrics,
        store_preds=run_config.store_preds,
        preds_dir=exp_dir / "preds",
    )
    try:
        result = target_function(config=configuration)
    except Exception as e:
        print(f"Error: {e}")
        result = TargetFunctionResult(
            None,
            TargetFunctionInfo(
                status=StatusType.CRASHED,
                train_loss=None,
                valid_loss=None,
                test_loss=None,
                average_split_fit_time=None,
                average_split_predict_time=None,
                average_split_predict_time_valid=None,
                average_split_predict_time_test=None,
                total_walltime=None,
                config=configuration.get_dictionary(),
                n_splits=None,
                train_metrics=None,
                valid_metrics=None,
                test_metrics=None,
                additional_info={"error": str(e), "traceback": traceback.format_exception(e)},
            ),
        )

    print(f"Result: {result}")

    additional_info = {
        **loader_additional_info,
        "git_hash": get_git_revision_short_hash(),
        "dataset_properties": dataset_properties,
    }
    # Save results
    results = RunResult(
        run_config=run_config,
        experiment_name=run_config.experiment_name,
        additional_info=additional_info,
        incumbent_configuation=result.info["config"],
        result=result,
    )

    results.to_json(exp_dir / "result.json")

    return results
