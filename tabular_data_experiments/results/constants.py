from __future__ import annotations

from enum import Enum

import openml

from tabular_data_experiments.utils.suites import CUSTOM_SUITES

# PLOTTING CONSTANTS
LABEL_NAMES = {
    "xgb": "XGB (EarlyStop, Train set, Hist)",
    "xgb_auto": "XGB (EarlyStop, Train set, Auto)",  # "XGB",
    "xgb_onehot": "XGB (OneHot, EarlyStop, Train set, Auto)",
    "rf": "RF",
    "reg_cocktails": "Cocktails",
    "catboost": "Catboost",
    "lgbm": "LGBM",
    "lgbm_logloss_early_stop_valid": "LGBM (EarlyStop, HPO set, LogLoss)",
    "lgbm_balanced_early_stop_valid": "LGBM (EarlyStop, HPO set, BalancedWeights)",
    "resnet": "ResNet",
    "ft_transformer": "FTTransformer",
    "saint": "SAINT",
    "linear": "Linear",
    "mlp": "MLP (rtdl)",
    "resnet_old": "ResNet_old",
    "resnet_extra_dim": "ResNet_extra_dim",
    "reg_cocktails_split_onehot": "Cocktails (Split)",
    "reg_cocktails_v1": "Cocktails (old)",
    "reg_cocktails_dim": "Cocktails (Embedding)",
    "reg_cocktails_combined": "Cocktails (SingleEmbedding)",
    "reg_cocktails_reduced": "Cocktails (Embedding-ReducedSearchSpace)",
    "hgbt": "HGBT",
    "sklearn_mlp": "MLP (sklearn)",
    "xgb_early_stop_valid": "XGB (EarlyStop, HPO set, Hist)",
    "xgb_onehot_early_stop_valid": "XGB (OneHot, EarlyStop, HPO set, Auto)",
    "catboost_early_stop_valid": "Catboost (EarlyStop, HPO set)",
    "lgbm_early_stop_valid": "LGBM (EarlyStop, HPO set)",
    "resnet_early_stop_valid": "ResNet (EarlyStop, HPO set)",
    "ft_transformer_early_stop_valid": "FTTransformer (EarlyStop, HPO set)",
    "saint_early_stop_valid": "SAINT (EarlyStop, HPO set)",
    "mlp_early_stop_valid": "MLP (EarlyStop, HPO set)",
    "tree": "DecisionTree",
}

FINAL_LABEL_NAMES = {
    "rf": "RF",
    "reg_cocktails": "Cocktails",
    "hgbt": "HGBT",
    "linear": "Linear",
    "sklearn_mlp": "MLP (sklearn)",
    "xgb_early_stop_valid": "XGB",
    "catboost_early_stop_valid": "Catboost",
    "lgbm_early_stop_valid": "LGBM",
    "resnet_early_stop_valid": "ResNet",
    "ft_transformer_early_stop_valid": "FTTransformer",
    "saint_early_stop_valid": "SAINT",
    "mlp_early_stop_valid": "MLP",
    "tree": "DecisionTree",
}

ALGORITHM_COLUMN_NAME = "method"
TASK_COLUMN_NAME = "Dataset"

CPU_MODELS = [
    "xgb",
    "xgb_onehot",
    "xgb_onehot_no_early_stop",
    "catboost",
    "lgbm",
    "hgbt",
    "xgb_early_stop_valid",
    "catboost_early_stop_valid",
    "lgbm_early_stop_valid",
    "linear",
    "rf",
    "sklearn_mlp",
]

BASELINE_MODELS = [
    "rf",
    "linear",
    "sklearn_mlp",
]

GPU_MODELS = [
    "reg_cocktails",
    "resnet",
    "ft_transformer",
    "saint",
    "mlp",
    "resnet_early_stop_valid",
    "ft_transformer_early_stop_valid",
    "saint_early_stop_valid",
    "mlp_early_stop_valid",
]

FINAL_MODELS = [
    "hgbt",
    "linear",
    "rf",
    "catboost_early_stop_valid",
    "xgb_early_stop_valid",
    "lgbm_early_stop_valid",
    "mlp_early_stop_valid",
    "resnet_early_stop_valid",
    "ft_transformer_early_stop_valid",
    "saint_early_stop_valid",
    "sklearn_mlp",
]

SHADES = {
    "red": ["#ff8080", "#ff4f4f", "#ff1e1e", "#ec0000", "#b00", "#8a0000", "#590000"],
    "blue": ["#5593c8", "#397bb3", "#2d618e", "#214868", "#152e43"],
}

CPU_COLOURS = {}
for model in CPU_MODELS:
    if model in FINAL_MODELS:
        CPU_COLOURS[model] = SHADES["red"][len(CPU_COLOURS)]
GPU_COLOURS = {}
for model in GPU_MODELS:
    if model in FINAL_MODELS:
        GPU_COLOURS[model] = SHADES["blue"][len(GPU_COLOURS)]
MODELS_TO_COLORS = {}
for model in FINAL_MODELS:
    MODELS_TO_COLORS[model] = CPU_COLOURS.get(model, GPU_COLOURS.get(model, "k"))

dataset_collections = {}

for dataset_collection_name, dataset_collection in CUSTOM_SUITES.items():
    dataset_collections[dataset_collection_name] = [str(task_id) for task_id in dataset_collection]
all_task_ids = dataset_collections["all_datasets"]
task_id_to_dataset_ids = {
    task_id: openml.tasks.get_task(task_id, download_data=False).dataset_id for task_id in all_task_ids
}


class MissingResultsHandler(Enum):
    """Enum for how to handle missing results"""

    drop_nan = "drop_nan"
    impute_nan = "fill_nan"
