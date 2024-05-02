"""Povides functionality to run experiments on tabular data."""
from __future__ import annotations

import time
from pathlib import Path

from ConfigSpace import Configuration

from tabular_data_experiments.data_loaders.base_loader import BaseLoader
from tabular_data_experiments.experiment_workflows.utils import (
    get_git_revision_short_hash,
)
from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.run_result_config import RunConfig, RunResult
from tabular_data_experiments.target_function.target_function import TargetFunction
from tabular_data_experiments.utils.data_utils import (
    get_required_dataset_info,
    split_data,
)


def run_refit_experiment(
    run_config: RunConfig,
    exp_dir: Path,
) -> RunResult:
    """
    Runs a refit experiment on a dataset.

    Args:
        run_config: Configuration for the experiment.
        exp_dir: Path to the experiment directory.
    """
    # Create experiment folder
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running experiment {run_config.experiment_name} in {exp_dir}.")
    # Create experiment log

    # Log experiment name
    # Load data
    time.time()
    data_loader = BaseLoader.create(run_config.data_loader, loader_kwargs={"task_id": run_config.task_id})
    # Here we have X, y, categorical_indicator, attribute_names
    X, y, categorical_indicator, attribute_names = data_loader.get_data()
    loader_additional_info = {"attribute_names": attribute_names}
    # Store attribute names to result dictionary and dataset attributes

    dataset_properties = get_required_dataset_info(X, y, categorical_indicator)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        fold_number=run_config.fold_number,
        task=data_loader.task if hasattr(data_loader, "task") else None,
    )

    config = None
    if isinstance(run_config.model_configuration, dict):
        config = Configuration(
            BaseModel.get_model_config_space(run_config.model_name, dataset_properties), run_config.model_configuration
        )
    elif run_config.model_configuration is not None:
        raise ValueError(
            f"Model configuration must be passed and must be a dictionary or"
            f"None (Default config is used) got {run_config.model_configuration}."
        )

    # Train best model on train data

    target_function = TargetFunction(
        X_train=X_train,
        y_train=y_train,
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
    result = target_function(config=config, seed=run_config.seed.get_seed("hpo"))
    print(f"Result: {result}")

    additional_info = {
        **loader_additional_info,
        "dataset_properties": dataset_properties,
        "git_hash": get_git_revision_short_hash(),
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
