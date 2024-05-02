"""
CMD_LINE Example:
python scripts/run_model_on_split.py --n_folds 2 --n_splits 2 --n_configs 1 --model_name 
reg_cocktails --task_ids 31 --exp_name Test --device cuda --n_parallel_jobs 1
--exp_dir <path_to_dir> --slurm_job_time_secs 3600 --partition <partition_name>
--seed 42 --overwrite

Run a model on `n_folds`, `n_splits` and `n_configs` on each task in a given list of
task ids, `task_ids` or a openml study `study` or a custom study defined
in `CUSTOM_SUITES` and passed via `custom_study` . Each variable is passed as a
command line argument. The experiment is run on a slurm cluster and the results
are stored in the `exp_dir` directory.
"""
from __future__ import annotations
import argparse
from itertools import product
import json
from pathlib import Path

import openml


from tabular_data_experiments.experiment_workflows.fit_single_split import (
    run_single_split_experiment,
)
from tabular_data_experiments.metrics import get_scorer
from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.target_function.utils import StatusType
from tabular_data_experiments.utils.suites import CUSTOM_SUITES
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.utils import ParallelJobDict, add_args_to_parser, add_cluster_args_to_parser, get_executer, get_split_experiment_dir, post_process_jobs

parser = argparse.ArgumentParser(description="Run a experiment with a given config")
args = [
    "--n_folds",
    "--task_ids",
    "--n_splits",
    "--run_config",
    "--n_configs",
    "--exp_dir",
    "--seed",
    "--manipulators",
    "--model_name",
    "--metric",
    "--data_loader",
    "--exp_name",
    "--device",
    "--additional_metrics",
    "--store_preds",
    "--study",
    "--custom_study",
    "--overwrite",
    "--n_workers",
    "--memory"
]
parser.add_argument("--cat_overwrite", action="store_true", help="Overwrite categorical features")
parser.add_argument(
    "--splits", type=int, nargs="+", help="splits to run", default=None
)
parser.add_argument(
    "--folds", type=int, nargs="+", help="folds to run", default=None
)
parser = add_args_to_parser(parser, args)
parser = add_cluster_args_to_parser(parser)

args = parser.parse_args()


if __name__ == "__main__":
    print(f"\n\n\nRunning experiment with args: {args}")
    slurm_executer = get_executer(
        partition=args.partition,
        log_folder=args.exp_dir / "logs",
        total_job_time_secs=args.slurm_job_time_secs,
        gpu=args.device != "cpu",
        cpu_nodes=1,
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
        if args.splits is not None:
            splits = args.splits
        else:
            splits = list(range(args.n_splits))
        if args.folds is not None:
            folds = args.folds
        else:
            folds = list(range(args.n_folds))
        for (
            task_id, fold_number, split_number, config_id
         ) in product(
            task_ids,
            folds,
            splits,
            list(range(1, args.n_configs+1))  # Config id starts at 1
        ):
            exp_dir = args.exp_dir
            exp_dir = get_split_experiment_dir(
                exp_dir=exp_dir,
                exp_name=args.exp_name,
                task_id=task_id,
                fold_number=fold_number,
                split_number=split_number,
                config_id=config_id,
            )
            if exp_dir.exists():
                # Check if the experiment was successful
                if (exp_dir / "result.json").exists() and not args.overwrite:
                    result = json.load((exp_dir / "result.json").open("r"))
                    has_cat = len(result["additional_info"]["dataset_properties"]["categorical_columns"]) > 0
                    was_success =  result['result'][1]['status'] == StatusType.SUCCESS
                    if was_success and not has_cat:
                        # if was success and not has cat, skip
                        print(f"Skipping task {task_id} on fold {fold_number} and split: {split_number} as it already exists")
                        print(f"Result: {result['result'][0]}")
                        continue
                    elif was_success and has_cat and args.cat_overwrite:
                        # if was_success but has cat and cat_overwrite, rerun
                        print(f"Found existing experiment dir {exp_dir} but has categorical features, rerunning experiment")
                    elif was_success and has_cat and not args.cat_overwrite:
                        # if was_success but has cat and not cat_overwrite, skip
                        print(f"Skipping task {task_id} on fold {fold_number} and split: {split_number} as it already exists")
                        print(f"Result: {result['result'][0]}")
                        continue
                    else:
                        # if not was success, skip
                        print(f"Skipping task {task_id} on fold {fold_number} and split: {split_number} as it timed out or failed")
                        continue
                        # print(f"Found existing experiment dir {exp_dir} but status is {result['result'][1]['status']}, rerunning experiment")
                elif args.overwrite:
                    # if overwrite, rerun
                    print(f"Found existing experiment dir {exp_dir} but got overwrite, rerunning experiment")

            exp_dir.mkdir(parents=True, exist_ok=True)

            seed = args.seed
            experiment_seed = ExperimentSeed(seed)

            if args.run_config is not None:
                run_config = RunConfig.from_yaml(args.run_config)
            else:
                model_seed = experiment_seed.get_seed("model")
                run_config = RunConfig(
                    experiment_name=args.exp_name,
                    seed=experiment_seed,
                    manipulators=args.manipulators,
                    metric=get_scorer(args.metric),
                    model_name=args.model_name,
                    fold_number=fold_number,
                    split_id=split_number,
                    config_id=config_id,
                    model_kwargs={
                        "seed": int(model_seed),
                        "device": args.device,
                    },
                    task_id=task_id,
                    additional_metrics=args.additional_metrics,
                    store_preds=args.store_preds,
                )
            run_config.to_yaml(exp_dir / "run_config.yaml")
            print(f"Submitting job for task {task_id} on fold {fold_number} and split: {split_number} with config id {config_id}")
            job = slurm_executer.submit(run_single_split_experiment, run_config=run_config, exp_dir=exp_dir)
            jobs.append(
                {
                    "job": job,
                    "exp_dir": exp_dir,
                    "task_id": task_id,
                    "fold_number": fold_number,
                    "split_number": split_number,
                    "config_id": config_id,
                    "custom_study": args.custom_study,
                })

    post_process_jobs(jobs)
