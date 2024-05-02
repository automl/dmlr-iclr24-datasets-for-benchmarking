from __future__ import annotations

from typing import Any, Callable

import json
import time
import warnings
from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import Constant
from scipy.stats.qmc import Sobol
from smac import HyperparameterOptimizationFacade, RunHistory, Scenario
from smac.initial_design import SobolInitialDesign
from smac.runhistory import TrialKey
from smac.runner.target_function_runner import TargetFunctionRunner

from tabular_data_experiments.optimizers.base_optimizer import AbstractOptimizer
from tabular_data_experiments.target_function.utils import TargetFunctionResult


class UnscrambledSobolInitialDesign(SobolInitialDesign):
    def _select_configurations(self) -> list[Configuration]:
        params = self._configspace.get_hyperparameters()

        constants = 0
        for p in params:
            if isinstance(p, Constant):
                constants += 1

        dim = len(params) - constants
        sobol_gen = Sobol(d=dim, scramble=False, seed=self._rng.randint(low=0, high=10000000))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sobol = sobol_gen.random(self._n_configs)

        return self._transform_continuous_designs(
            design=sobol, origin="Initial Design: Sobol", configspace=self._configspace
        )


class SOBOLOptimizer(AbstractOptimizer):
    def __init__(
        self,
        target_function: Callable,
        config_space: ConfigurationSpace,
        n_trials: int,
        name: str,
        output_directory: Path,
        seed: int,
        n_workers: int = 1,
    ) -> None:
        super().__init__(target_function=target_function, config_space=config_space)
        self.scenario = Scenario(
            self.config_space, name=name, n_trials=n_trials - 1, seed=seed, output_directory=output_directory
        )
        self.run_history = RunHistory()
        self.design = UnscrambledSobolInitialDesign(
            self.scenario, n_configs=self.scenario.n_trials, seed=seed, max_ratio=1.0
        )
        if self.scenario.output_directory.exists():
            print(f"Output directory {self.scenario.output_directory} already exists")
            self.run_history.load(self.scenario.output_directory / "runhistory.json", self.config_space)
            self.ids_config_all = json.load((self.scenario.output_directory / "configurations.json").open("r"))
            print(f"Continuing previous run")
        else:
            self.scenario.output_directory.mkdir(parents=True, exist_ok=True)

            configurations_to_run: list[Configuration] = [
                self.config_space.get_default_configuration(),
                *self.design.select_configurations(),
            ]
            self.ids_config_all = {i: config.get_dictionary() for i, config in enumerate(configurations_to_run)}

        self.runner = TargetFunctionRunner(
            scenario=self.scenario,
            target_function=self.target_function,
        )

        json.dump(self.ids_config_all, (self.scenario.output_directory / "configurations.json").open("w"))

    def optimize(self) -> Configuration:
        """Runs the configurations sampled by sobol sequence."""
        for i, config in self.ids_config_all.items():
            configuration = Configuration(self.config_space, values=config)
            trial = int(i) + 1
            if trial in self.run_history.ids_config:
                print(f"Skipping trial {trial}/{int(self.scenario.n_trials)+1} as it has already been run")
                continue
            print(f"Running trial {trial}/{int(self.scenario.n_trials)+1}")
            start_time = time.time()
            status, cost, runtime, additional_info = self.runner.run(config=configuration)
            end_time = time.time()
            self.run_history.add(
                config=configuration,
                cost=cost,
                status=status,
                time=runtime,
                additional_info=additional_info,
                starttime=start_time,
                endtime=end_time,
            )
            print(f"Trial {trial}/{int(self.scenario.n_trials)+1} finished with status {status} and cost {cost}")
            self.run_history.save(self.scenario.output_directory / "runhistory.json")

        return self.get_incumbent_configuration(self.run_history)

    def get_incumbent_result(self) -> TargetFunctionResult:
        trial_value = self.run_history[self._get_incumbent_trial_key(self.run_history)]
        result = TargetFunctionResult(loss=trial_value.cost, info=trial_value.additional_info)
        return result
