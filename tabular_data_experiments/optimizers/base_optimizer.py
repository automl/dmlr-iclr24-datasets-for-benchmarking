from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Type

from ConfigSpace import Configuration, ConfigurationSpace
from smac import RunHistory
from smac.runhistory import TrialKey

from tabular_data_experiments.target_function.utils import TargetFunctionResult


class AbstractOptimizer(ABC):
    def __init__(self, target_function: Callable, config_space: ConfigurationSpace) -> None:
        super().__init__()
        self.target_function = target_function
        self.config_space = config_space

    @staticmethod
    def _get_incumbent_trial_key(runhistory: RunHistory) -> TrialKey:
        return sorted(runhistory, key=lambda r: runhistory[r].cost)[0]

    @staticmethod
    def get_incumbent_configuration(runhistory: RunHistory) -> Configuration:
        """Get the best configuration from the runhistory"""
        return runhistory.ids_config[AbstractOptimizer._get_incumbent_trial_key(runhistory).config_id]

    @abstractmethod
    def optimize(self, **kwargs) -> Configuration:
        pass

    @abstractmethod
    def get_incumbent_result(self) -> TargetFunctionResult | None:
        pass

    @classmethod
    def get_optimizer(cls, optimizer_name: str) -> Type[AbstractOptimizer]:
        if optimizer_name == "hpo":
            from tabular_data_experiments.optimizers.hpo import HPOOptimizer

            return HPOOptimizer
        elif optimizer_name == "sobol":
            from tabular_data_experiments.optimizers.sobol import SOBOLOptimizer

            return SOBOLOptimizer
        elif optimizer_name == "default":
            from tabular_data_experiments.optimizers.default import DefaultOptimizer

            return DefaultOptimizer
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")
