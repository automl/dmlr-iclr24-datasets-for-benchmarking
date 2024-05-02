from __future__ import annotations

from typing import Any, Callable

from ConfigSpace import Configuration, ConfigurationSpace
from smac import HyperparameterOptimizationFacade, Scenario

from tabular_data_experiments.optimizers.base_optimizer import AbstractOptimizer
from tabular_data_experiments.target_function.utils import TargetFunctionResult


class HPOOptimizer(AbstractOptimizer):
    def __init__(
        self,
        target_function: Callable,
        config_space: ConfigurationSpace,
        scenario_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(target_function=target_function, config_space=config_space)
        scenario = Scenario(self.config_space, **scenario_kwargs)
        self.smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=self.target_function.__call__,
        )

    def optimize(self) -> Configuration:
        return self.smac.optimize()

    def get_incumbent_result(self) -> TargetFunctionResult:
        trial_value = self.smac.runhistory[self._get_incumbent_trial_key(self.smac.runhistory)]
        result = TargetFunctionResult(loss=trial_value.cost, info=trial_value.additional_info)
        return result
