"""Run and result configuration classes."""
from __future__ import annotations

from typing import Any, Generic, Literal

import dataclasses
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

import numpy as np
import yaml

from tabular_data_experiments.metrics import Scorer
from tabular_data_experiments.target_function.utils import TargetFunctionResult
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.types import PathType
from tabular_data_experiments.utils.utils import dict_repr


class DataclassEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, "to_json"):
            return obj.to_json()
        # TODO: This is a hack to get around the fact that
        # the metric is not json serialisable
        # After fixing metric calculation, take a look.
        if hasattr(obj, "_score_func"):
            return obj._score_func.__name__
        if callable(obj):
            return obj.__name__
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj.absolute())
        return super().default(obj)


class YamlDataClass(Generic[PathType]):
    """Base class for data classes that can be serialized to and from YAML."""

    @classmethod
    def from_yaml(cls, file_path: PathType) -> YamlDataClass:
        """Load a data class from a YAML file.

        Args:
            file_path: _description_

        Returns:
            _description_
        """
        with open(file_path, "r") as f:
            data = yaml.unsafe_load(f)
        return cls(**data)

    def to_yaml(self, file_path: PathType) -> None:
        """Save a data class to a YAML file."""
        with open(file_path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_dict(self) -> dict[str, Any]:
        """Convert a data class to a dictionary."""
        assert dataclasses.is_dataclass(self), "Required to be a dataclass."
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self, file_path: PathType) -> None:
        """Convert a data class to a JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4, cls=DataclassEncoder)
        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({dict_repr(self.to_dict())})"


@dataclass
class RunConfig(YamlDataClass):
    """Run configuration."""

    experiment_name: str
    seed: ExperimentSeed
    manipulators: list[str]
    metric: Scorer
    model_name: str
    task_id: int
    fold_number: int | None = None
    optimizer: Literal["hpo", "sobol", "default"] = "default"
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    data_loader: str = "openml"
    manipulator_kwargs: dict[str, Any] = field(default_factory=dict)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    splitter: Any | None = None
    splitter_kwargs: dict[str, Any] = field(default_factory=dict)
    model_configuration: dict[str, Any] | None = None
    additional_metrics: list[str] | None = None
    store_preds: bool = False
    split_id: int | None = None
    config_id: int | None = None


@dataclass
class RunResult(YamlDataClass):
    """Result of a run."""

    run_config: RunConfig
    experiment_name: str
    additional_info: dict[str, Any]
    incumbent_configuation: dict[str, Any]
    result: TargetFunctionResult
