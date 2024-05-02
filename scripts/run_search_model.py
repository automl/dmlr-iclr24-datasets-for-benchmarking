"""Script to search for the best model hyperparameters for a given dataset."""
from __future__ import annotations
import argparse

from sklearn.model_selection import StratifiedKFold
from tabular_data_experiments.metrics import get_scorer

from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.experiment_workflows.refit_experiment import run_refit_experiment
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.utils import (
    ParallelJobDict,
    add_cluster_args_to_parser,
    add_common_args_to_parser,
    add_search_args_to_parser,
    get_executer,
    get_experiment_dir,
    post_process_jobs,
)


parser = argparse.ArgumentParser(description="Run a experiment with a given config")
parser.add_argument("--task_id", type=int, help="Task id to run", default=31)
parser = add_common_args_to_parser(parser)
parser = add_cluster_args_to_parser(parser)
parser = add_search_args_to_parser(parser)

args = parser.parse_args()

if __name__ == "__main__":
    slurm_executer = get_executer(
        partition=args.partition,
        log_folder=args.exp_dir / "logs",
        total_job_time_secs=args.slurm_job_time_secs,
        gpu=args.device != "cpu",
        cpu_nodes=args.n_workers,
        slurm_max_parallel_jobs=args.n_parallel_jobs,
    )

    task_id = args.task_id

    jobs: list[ParallelJobDict] = []
    with slurm_executer.batch():
        for fold_number in range(args.n_folds):
            exp_dir = get_experiment_dir(args.exp_dir, args.exp_name, task_id, fold_number)
            exp_dir.mkdir(parents=True, exist_ok=True)

            experiment_seed = ExperimentSeed(args.seed)

            if args.run_config is not None:
                run_config: RunConfig = RunConfig.from_yaml(args.run_config)
                run_config.fold_number = fold_number  # update fold number
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
                    "hpo": True,
                    "hpo_kwargs": {
                        "scenario_kwargs": {
                            "seed": int(hpo_seed),
                            "output_directory": exp_dir / "hpo",
                            "deterministic": True,
                            "name": args.exp_name,
                            "n_trials": args.n_trials,
                            "n_workers": args.n_workers,
                        },
                    },
                    "fold_number": fold_number,
                    "model_kwargs": {"seed": int(model_seed), "device": args.device},
                    "splitter": StratifiedKFold,
                    "splitter_kwargs": {"n_splits": 2, "random_state": int(data_seed), "shuffle": True},
                    "store_preds": args.store_preds,
                }
                run_config = RunConfig(
                    task_id=task_id,
                    **run_config_kwargs,
                )
            run_config.to_yaml(exp_dir / "run_config.yaml")
            print(f"Running task {task_id} on fold {fold_number}")
            job = slurm_executer.submit(run_refit_experiment, run_config=run_config, exp_dir=exp_dir)
            jobs.append({"job": job, "exp_dir": exp_dir, "fold_number": fold_number, "task_id": task_id})

    post_process_jobs(jobs)
