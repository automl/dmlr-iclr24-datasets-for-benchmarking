"""Script to run passed configuration of model on each dataset paralellised on cluster."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path

import openml

from tabular_data_experiments.metrics import get_scorer

from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.experiment_workflows.refit_experiment import run_refit_experiment
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.utils import (
    ParallelJobDict,
    add_cluster_args_to_parser,
    add_common_args_to_parser,
    get_executer,
    get_experiment_dir,
    post_process_jobs,
)


parser = argparse.ArgumentParser(description="Run a experiment with a given config")

parser.add_argument("--config", type=Path, help="Path to configuration file of the model")
parser.add_argument("--task_ids", type=int, nargs="+", help="Task ids to run", default=[31])
parser.add_argument("--study", type=int, help="OpenML Suite id")
parser = add_common_args_to_parser(parser)
parser = add_cluster_args_to_parser(parser)

args = parser.parse_args()

if __name__ == "__main__":
    slurm_executer = get_executer(
        partition=args.partition,
        log_folder=args.exp_dir / "logs",
        total_job_time_secs=args.slurm_job_time_secs,
        gpu=args.device != "cpu",
        slurm_max_parallel_jobs=args.n_parallel_jobs,
    )

    jobs: list[ParallelJobDict] = []

    task_ids = args.task_ids
    if args.study is not None:
        study = openml.study.get_suite(args.study)
        task_ids = study.tasks

    with slurm_executer.batch():
        for task_id, fold_number in product(task_ids, list(range(args.n_folds))):
            exp_dir = get_experiment_dir(args.exp_dir, args.exp_name, task_id, fold_number)
            exp_dir.mkdir(parents=True, exist_ok=True)

            experiment_seed = ExperimentSeed(args.seed)

            if args.run_config is not None:
                run_config: RunConfig = RunConfig.from_yaml(args.run_config)
                run_config.task_id = task_id
                run_config.fold_number = fold_number
            else:
                model_seed, data_seed, hpo_seed = (
                    experiment_seed.get_seed("model"),
                    experiment_seed.get_seed("data"),
                    experiment_seed.get_seed("hpo"),
                )
                run_config_kwargs = {
                    "experiment_name": args.exp_name,
                    "seed": experiment_seed,
                    "manipulators": args.manipulators,
                    "metric": get_scorer(args.metric),
                    "model_name": args.model_name,
                    "fold_number": fold_number,
                    "model_kwargs": {"seed": int(model_seed), "device": args.device},
                    "manipulator_kwargs": {
                        "remove_missing_values__threshold": 0.3,
                        "remove_high_cardinality__threshold": 10,
                    },
                    "store_preds": args.store_preds,
                }
                run_config = RunConfig(
                    task_id=task_id,
                    **run_config_kwargs,
                )

            if args.config is not None:
                run_config.model_configuration = json.load(args.config.open())
            run_config.to_yaml(exp_dir / "run_config.yaml")
            print(f"Running task {task_id} on fold {fold_number}")
            job = slurm_executer.submit(run_refit_experiment, run_config=run_config, exp_dir=exp_dir)
            jobs.append({"job": job, "exp_dir": exp_dir, "task_id": task_id, "fold_number": fold_number})

    post_process_jobs(jobs)
