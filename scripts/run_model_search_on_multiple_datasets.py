"""Script to run default configuration of model on each dataset paralellised on cluster."""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import openml

from sklearn.model_selection import StratifiedKFold
from tabular_data_experiments.metrics import get_scorer

from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.experiment_workflows.refit_experiment import run_refit_experiment
from tabular_data_experiments.utils.suites import CUSTOM_SUITES
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
parser.add_argument("--task_ids", type=int, nargs="+", help="Task ids to run", default=[31])
parser.add_argument("--study", type=int, help="OpenML Suite id")
parser.add_argument("--custom_study", type=str, help="Name of custom study")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
parser = add_common_args_to_parser(parser)
parser = add_cluster_args_to_parser(parser)
parser = add_search_args_to_parser(parser)


args = parser.parse_args()


def check_all_errors(exp_dir: Path) -> bool:
    """Check if all runs in the runhistory have errors"""
    runhistory_files = list(exp_dir.glob("**/runhistory.json"))
    if len(runhistory_files) == 0:
        return False
    else:
        print(f"Found {len(runhistory_files)} runhistory files")
        runhistory_file = runhistory_files[0]
    runhistory = json.load(open(runhistory_file, "r"))
    data = runhistory["data"]
    all_errors = True
    for run in data:
        if "error" not in run[-1]:
            all_errors = False
            break
    return all_errors


if __name__ == "__main__":
    print(f"\n\n\nRunning experiment with args: {args}")
    slurm_executer = get_executer(
        partition=args.partition,
        log_folder=args.exp_dir / "logs",
        total_job_time_secs=args.slurm_job_time_secs,
        gpu=args.device != "cpu",
        cpu_nodes=args.n_workers,
        slurm_max_parallel_jobs=args.n_parallel_jobs,
    )

    jobs: list[ParallelJobDict] = []

    task_ids = args.task_ids
    if args.study is not None:
        assert args.custom_study is None, "Cannot specify both study and custom_study"
        study = openml.study.get_suite(args.study)
        task_ids = study.tasks

    if args.custom_study is not None:
        assert args.study is None, "Cannot specify both study and custom_study"
        task_ids = CUSTOM_SUITES[args.custom_study]

    with slurm_executer.batch():
        for task_id, fold_number in product(task_ids, list(range(args.n_folds))):
            exp_dir = get_experiment_dir(args.exp_dir, args.exp_name, task_id, fold_number)
            if exp_dir.exists() and not args.overwrite:
                if (exp_dir / "result.json").exists():
                    result = json.load((exp_dir / "result.json").open("r"))
                    if "error" not in result["additional_info"]["incumbent_result"]:
                        print(f"Skipping {exp_dir} as it already has results")
                        continue
                    else:
                        shutil.rmtree(exp_dir)
                else:
                    # check if previous runs were all errors
                    if check_all_errors(exp_dir):
                        print(f"Rerunning {exp_dir} as previous runs were all errors")
                        shutil.rmtree(exp_dir)

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
                    "additional_metrics": args.additional_metrics,
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
                    "splitter": StratifiedKFold,
                    "splitter_kwargs": {"n_splits": 5, "random_state": int(data_seed), "shuffle": True},
                    "store_preds": args.store_preds,
                }
                run_config = RunConfig(
                    task_id=task_id,
                    **run_config_kwargs,
                )
            run_config.to_yaml(exp_dir / "run_config.yaml")
            print(f"Submitting job for task {task_id} on fold {fold_number}")
            job = slurm_executer.submit(run_refit_experiment, run_config=run_config, exp_dir=exp_dir)
            jobs.append({"job": job, "exp_dir": exp_dir, "task_id": task_id, "fold_number": fold_number})

    post_process_jobs(jobs)
