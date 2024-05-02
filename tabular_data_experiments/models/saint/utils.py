"""Get the SAINT config."""
from __future__ import annotations

from typing import Any, Iterable

import uuid
from pathlib import Path

import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_saint_config(
    config: dict[str, Any],
    categorical_indicator: Iterable[bool],
    cat_dims,
    num_features: int,
    epochs: int = 100,
    early_stopping_rounds: int = 10,
    output_dir: Path = Path("output"),
    device: str = "cpu",
    model_name: str = "saint",
    num_classes: int = 1,
    data_parallel: bool = False,
    objective: str = "binary",
):
    """Get the SAINT config."""
    args_dic = {}
    params_dic = {}
    for key in config.keys():
        if key.startswith("args__"):
            args_dic[key.replace("args__", "")] = config[key]
        elif key.startswith("params__"):
            params_dic[key.replace("params__", "")] = config[key]
    args_dic["num_classes"] = num_classes
    args_dic["model_name"] = model_name
    args_dic["data_parallel"] = data_parallel
    args_dic["model_id"] = uuid.uuid4().int
    args_dic["cat_idx"] = np.where(categorical_indicator)[0]
    args_dic["cat_dims"] = cat_dims
    args_dic["num_features"] = num_features
    args_dic["gpu_ids"] = [0]
    args_dic["use_gpu"] = device != "cpu"
    args_dic["output_dir"] = output_dir
    args_dic["epochs"] = epochs
    args_dic["early_stopping_rounds"] = early_stopping_rounds
    args_dic["objective"] = objective
    args = AttrDict()
    params = AttrDict()
    args.update(args_dic)
    params.update(params_dic)
    return args, params
