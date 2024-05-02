"""Utility functions for the regularization cocktails"""
from __future__ import annotations

from typing import Any, Literal

import logging.handlers

from autoPyTorch.datasets.tabular_dataset import TabularDataset
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates,
)

EMBEDDING_CHOICES_NEW = ("LearnedEntityEmbedding", "CombinedEmbedding", "LearnedEntityEmbeddingReduced")
EMBEDDING_CHOICES_OLD = ("LearnedEntityEmbedding",)

embedding_choice_to_component_name = {
    "none": "NoEmbedding",
    "reduced": "LearnedEntityEmbeddingReduced",
    "dim_reduce": "LearnedEntityEmbedding",
    "combined": "CombinedEmbedding",
}


def get_updates_for_regularization_cocktails(
    search_embedding: bool = False,
    embedding_choice: Literal["none", "reduced", "dim_reduce", "combined"] = "none",
    split_min_categories: bool = False,
    fixed_other_params: bool = True,
    new_code: bool = True,
    has_categorical_columns: bool = True,
) -> HyperparameterSearchSpaceUpdates:
    """
    These updates replicate the regularization cocktail paper search space.
    Args:
    Returns:
    ________
        search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """

    search_space_updates = HyperparameterSearchSpaceUpdates()

    if not fixed_other_params:
        get_other_search_params(search_space_updates)
    else:
        get_fixed_search_params(search_space_updates)

    # network
    search_space_updates.append(
        node_name="network_init",
        hyperparameter="__choice__",
        value_range=["NoInit"],
        default_value="NoInit",
    )
    # architecture head
    search_space_updates.append(
        node_name="network_head",
        hyperparameter="__choice__",
        value_range=["no_head"],
        default_value="no_head",
    )
    search_space_updates.append(
        node_name="network_head",
        hyperparameter="no_head:activation",
        value_range=["relu"],
        default_value="relu",
    )

    if not has_categorical_columns:
        search_space_updates.append(
            node_name="network_embedding",
            hyperparameter="__choice__",
            value_range=[embedding_choice_to_component_name["none"]],
            default_value=embedding_choice_to_component_name["none"],
        )
    # network embedding
    elif not search_embedding:
        search_space_updates.append(
            node_name="network_embedding",
            hyperparameter="__choice__",
            value_range=[embedding_choice_to_component_name[embedding_choice]],
            default_value=embedding_choice_to_component_name[embedding_choice],
        )
    else:
        embedding_choices = EMBEDDING_CHOICES_NEW if new_code else EMBEDDING_CHOICES_OLD
        embedding_choices = list(embedding_choices)
        embedding_choices.extend(["NoEmbedding"])
        search_space_updates.append(
            node_name="network_embedding",
            hyperparameter="__choice__",
            value_range=embedding_choices,
            default_value=embedding_choice_to_component_name[embedding_choice],
        )

    if not split_min_categories and new_code:
        search_space_updates.append(
            node_name="column_splitter",
            hyperparameter="min_categories_for_embedding",
            value_range=[0],
            default_value=0,
        )

    # training updates
    search_space_updates.append(
        node_name="lr_scheduler",
        hyperparameter="__choice__",
        value_range=["CosineAnnealingWarmRestarts"],
        default_value="CosineAnnealingWarmRestarts",
    )
    search_space_updates.append(
        node_name="lr_scheduler",
        hyperparameter="CosineAnnealingWarmRestarts:n_restarts",
        value_range=[3],
        default_value=3,
    )
    # optimizer
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="AdamWOptimizer:beta1",
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="AdamWOptimizer:beta2",
        value_range=[0.999],
        default_value=0.999,
    )

    # preprocessing
    search_space_updates.append(
        node_name="feature_preprocessor",
        hyperparameter="__choice__",
        value_range=["NoFeaturePreprocessor"],
        default_value="NoFeaturePreprocessor",
    )

    # set preprocessing config to None
    # Note: These changes have no affect on the pipeline
    # since the pipeline does not preprocess the data
    # it assumes that the data is already preprocessed
    # in the input data validator and by the user
    search_space_updates.append(
        node_name="encoder",
        hyperparameter="__choice__",
        value_range=["NoEncoder"],
        default_value="NoEncoder",
    )
    search_space_updates.append(
        node_name="scaler",
        hyperparameter="__choice__",
        value_range=["NoScaler"],
        default_value="NoScaler",
    )

    return search_space_updates


def get_other_search_params(search_space_updates):
    search_space_updates.append(
        node_name="network_init",
        hyperparameter="NoInit:bias_strategy",
        value_range=["Zero"],
        default_value="Zero",
    )

    # backbone architecture choices
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="__choice__",
        value_range=["ShapedResNetBackbone", "ShapedMLPBackbone"],
        default_value="ShapedResNetBackbone",
    )

    # resnet backbone
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:resnet_shape",
        value_range=["funnel"],
        default_value="funnel",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:dropout_shape",
        value_range=["funnel"],
        default_value="funnel",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:max_dropout",
        value_range=[0, 1],
        default_value=0.5,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:num_groups",
        value_range=[1, 4],
        default_value=2,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:blocks_per_group",
        value_range=[1, 3],
        default_value=2,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:output_dim",
        value_range=[32, 512],
        default_value=64,
        log=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:max_units",
        value_range=[32, 512],
        default_value=64,
        log=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:activation",
        value_range=["relu"],
        default_value="relu",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:use_skip_connection",
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:use_batch_norm",
        value_range=[True],
        default_value=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:shake_shake_update_func",
        value_range=["shake-shake"],
        default_value="shake-shake",
    )
    # mlp backbone
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedMLPBackbone:mlp_shape",
        value_range=["funnel"],
        default_value="funnel",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedMLPBackbone:num_groups",
        value_range=[1, 5],
        default_value=2,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedMLPBackbone:output_dim",
        value_range=[64, 1024],
        default_value=64,
        log=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedMLPBackbone:max_units",
        value_range=[64, 1024],
        default_value=64,
        log=True,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedMLPBackbone:activation",
        value_range=["relu"],
        default_value="relu",
    )

    # optimizer
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="__choice__",
        value_range=["AdamOptimizer", "SGDOptimizer"],
        default_value="AdamOptimizer",
    )
    # adam
    search_space_updates.append(
        node_name="optimizer", hyperparameter="AdamOptimizer:lr", value_range=[1e-4, 1e-1], default_value=1e-3, log=True
    )

    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="AdamOptimizer:weight_decay",
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )

    # sgd
    search_space_updates.append(
        node_name="optimizer", hyperparameter="SGDOptimizer:lr", value_range=[1e-4, 1e-1], default_value=1e-3, log=True
    )

    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="SGDOptimizer:weight_decay",
        value_range=[1e-5, 1e-1],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="SGDOptimizer:momentum",
        value_range=[0.1, 0.999],
        default_value=0.1,
        log=True,
    )
    search_space_updates.append(
        node_name="data_loader", hyperparameter="batch_size", value_range=[16, 512], default_value=128, log=True
    )


def get_fixed_search_params(search_space_updates):
    """Returns the fixed search parameters for the given search space updates"""
    search_space_updates.append(
        node_name="data_loader",
        hyperparameter="batch_size",
        value_range=[128],
        default_value=128,
    )
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="AdamWOptimizer:lr",
        value_range=[1e-3],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name="optimizer",
        hyperparameter="__choice__",
        value_range=["AdamWOptimizer"],
        default_value="AdamWOptimizer",
    )

    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:shake_shake_update_func",
        value_range=["even-even"],
        default_value="even-even",
    )
    # backbone architecture
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="__choice__",
        value_range=["ShapedResNetBackbone"],
        default_value="ShapedResNetBackbone",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:resnet_shape",
        value_range=["brick"],
        default_value="brick",
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:num_groups",
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:blocks_per_group",
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:output_dim",
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:max_units",
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name="network_backbone",
        hyperparameter="ShapedResNetBackbone:activation",
        value_range=["relu"],
        default_value="relu",
    )


def get_pipeline_options(
    budget: int = 105,
    budget_type: str = "epochs",
    device: str = "cpu",
    early_stopping: int = -1,
) -> dict[str, Any]:
    """Returns the pipeline options for the given budget and budget type"""
    return {
        "device": device,
        "budget_type": budget_type,
        "min_epochs": budget,
        "epochs": budget,
        "torch_num_threads": 1,
        "early_stopping": early_stopping,
        "use_tensorboard_logger": "False",
        "metrics_during_training": False,
    }


def init_fit_dictionary(
    dataset_properties: dict[str, Any],
    budget_type: str,
    budget: int,
    metric_name: str,
    backend,
    dataset: TabularDataset,
    num_run: int = 0,
    device: str = "cpu",
    early_stopping: int = -1,
) -> dict[str, Any]:
    """
    Initialises the fit dictionary

    Args:
        logger_port (int):
            Logging is performed using a socket-server scheme to be robust against many
            parallel entities that want to write to the same file. This integer states the
            socket port for the communication channel.
        pipeline_config (Dict[str, Any]):
            Defines the content of the pipeline being evaluated. For example, it
            contains pipeline specific settings like logging name, or whether or not
            to use tensorboard.
    Returns:
        None
    """

    fit_dictionary: dict[str, Any] = {"dataset_properties": dataset_properties}

    split_id = 0
    train_indices, val_indices = dataset.splits[split_id]

    fit_dictionary.update(
        {
            "X_train": dataset.train_tensors[0],
            "y_train": dataset.train_tensors[1],
            "X_test": dataset.test_tensors[0],
            "y_test": dataset.test_tensors[1],
            "backend": backend,
            "logger_port": logging.handlers.DEFAULT_TCP_LOGGING_PORT,
            "optimize_metric": metric_name,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "split_id": split_id,
            "num_run": num_run,
        }
    )

    pipeline_config = get_pipeline_options(budget, budget_type, device=device, early_stopping=early_stopping)
    fit_dictionary.update(pipeline_config)
    return fit_dictionary
