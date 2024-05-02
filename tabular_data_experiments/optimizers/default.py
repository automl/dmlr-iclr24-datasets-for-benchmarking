from typing import Callable

from ConfigSpace import Configuration, ConfigurationSpace

from tabular_data_experiments.optimizers.base_optimizer import AbstractOptimizer


class DefaultOptimizer(AbstractOptimizer):
    def __init__(self, target_function: Callable, config_space: ConfigurationSpace) -> None:
        super().__init__(target_function=target_function, config_space=config_space)

    def optimize(self) -> Configuration:
        return self.config_space.get_default_configuration()

    def get_incumbent_result(self) -> None:
        return None
