"""Utility functions."""
from __future__ import annotations

from typing import Any, Iterator, List, Type

import argparse
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from submitit import AutoExecutor, Executor, Job, SlurmExecutor
from submitit.core import core
from submitit.core.utils import DelayedSubmission, JobPaths
from submitit.slurm.slurm import SlurmJob
from typing_extensions import TypedDict

HYPERPARAMETER_NAME_TO_TABLE_NAME = {
    "UniformIntegerHyperparameter": "UniformInt",
    "UniformFloatHyperparameter": "UniformFloat",
    "CategoricalHyperparameter": "Categorical",
    "Constant": "Categorical",
    "NormalIntegerHyperparameter": "NormalInt",
    "NormalFloatHyperparameter": "NormalFloat",
}  # noqa


def dict_repr(d: dict[Any, Any] | None) -> str:
    """Display long message in dict as it is."""
    if isinstance(d, dict):
        return "\n".join(["{}: {}".format(k, v if not isinstance(v, dict) else dict_repr(v)) for k, v in d.items()])
    else:
        return "None"


def chunks(lst: list[Any], n: int) -> Iterator[Any]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def count_parameters(model: torch.nn.Module) -> int:
    """
    A method to get the trainable parameter count from the model

    Args:
        model (torch.nn.Module): the module from which to count parameters

    Returns:
        trainable_parameter_count: only the parameters being optimized
    """
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_parameter_count


class BoschSlurmExecutor(SlurmExecutor):
    """Slurm executor to submit jobs to the Bosch cluster."""

    def _make_submission_command(self, submission_file_path):
        return ["sbatch", str(submission_file_path), "--bosch"]

    def _internal_process_submissions(self, delayed_submissions: List[DelayedSubmission]) -> List[core.Job[Any]]:
        if len(delayed_submissions) == 1:
            return super()._internal_process_submissions(delayed_submissions)
        # array
        folder = JobPaths.get_first_id_independent_folder(self.folder)
        folder.mkdir(parents=True, exist_ok=True)
        timeout_min = self.parameters.get("time", 5)
        pickle_paths = []
        for d in delayed_submissions:
            pickle_path = folder / f"{uuid.uuid4().hex}.pkl"
            d.set_timeout(timeout_min, self.max_num_timeout)
            d.dump(pickle_path)
            pickle_paths.append(pickle_path)
        n = len(delayed_submissions)
        # Make a copy of the executor, since we don't want other jobs to be
        # scheduled as arrays.
        array_ex = BoschSlurmExecutor(self.folder, self.max_num_timeout)
        array_ex.update_parameters(**self.parameters)
        array_ex.parameters["map_count"] = n
        self._throttle()

        first_job: core.Job[Any] = array_ex._submit_command(self._submitit_command_str)
        tasks_ids = list(range(first_job.num_tasks))
        jobs: List[core.Job[Any]] = [
            SlurmJob(folder=self.folder, job_id=f"{first_job.job_id}_{a}", tasks=tasks_ids) for a in range(n)
        ]
        for job, pickle_path in zip(jobs, pickle_paths):
            job.paths.move_temporary_file(pickle_path, "submitted_pickle")
        return jobs


PARTITION_TO_EXECUTER: dict[str, Type[Executor]] = {"bosch": BoschSlurmExecutor, "other": AutoExecutor}


def get_executer_class(partition: str) -> Type[Executor]:
    """Get the executor class for the given partition."""
    if "bosch" in partition:
        key = "bosch"
    else:
        key = "other"
    return PARTITION_TO_EXECUTER[key]


def get_executer_params(
    timeout: float,
    partition: str,
    gpu: bool = False,
    cpu_nodes: int = 1,
    mem_per_cpu: int = 2000,
    slurm_max_parallel_jobs: int = 20,
) -> dict[str, Any]:
    """Get the executor parameters for the given partition."""
    if gpu:
        if "bosch" in partition:
            return {
                "time": int(timeout),
                "partition": partition,
                "gres": f"gpu:{cpu_nodes}",
                "array_parallelism": slurm_max_parallel_jobs,
            }  # , 'gpus': cpu_nodes} #, 'nodes': 1} #, 'cpus_per_task': cpu_nodes}
        else:
            return {
                "timeout_min": int(timeout),
                "slurm_partition": partition,
                "slurm_gres": f"gpu:{cpu_nodes}",
                "slurm_array_parallelism": slurm_max_parallel_jobs,
            }  # , 'slurm_num_gpus': cpu_nodes} #slurm_gpus_per_task': cpu_nodes} #  'slurm_tasks_per_node': 1,
    else:
        return {
            "time": int(timeout),
            "partition": partition,
            "mem_per_cpu": mem_per_cpu,
            "nodes": 1,
            "cpus_per_task": cpu_nodes,
            "array_parallelism": slurm_max_parallel_jobs,
        }


def get_executer(
    partition: str,
    log_folder: str,
    gpu: bool = False,
    slurm_max_parallel_jobs: int = 20,
    total_job_time_secs: float = 3600,
    cpu_nodes: int = 1,
    mem_per_cpu: int = 12000,
) -> Executor:
    """Get the executor for the given partition."""
    slurm_executer = get_executer_class(partition)(folder=log_folder)
    slurm_executer.update_parameters(
        **get_executer_params(
            np.ceil(total_job_time_secs / 60),
            partition,
            gpu,
            cpu_nodes=cpu_nodes,
            mem_per_cpu=mem_per_cpu,
            slurm_max_parallel_jobs=slurm_max_parallel_jobs,
        )
    )
    return slurm_executer


def set_seed(seed: int):
    # Setting up reproducibility
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_experiment_dir(exp_dir: Path, exp_name: str, task_id: int, fold_number: int) -> Path:
    """Get the experiment directory for the given task and fold."""
    return exp_dir / exp_name / f"task_{task_id}" / f"fold_{fold_number}"


def get_split_experiment_dir(
    exp_dir: Path, exp_name: str, task_id: int, fold_number: int, split_number: int, config_id: int
) -> Path:
    """Get the experiment directory for the given split."""
    return (
        get_experiment_dir(exp_dir=exp_dir, exp_name=exp_name, task_id=task_id, fold_number=fold_number)
        / f"split_{split_number}"
        / f"config_{config_id}"
    )


class ParallelJobDict(TypedDict):
    """Dict to store the jobs for each fold and task."""

    job: Job
    exp_dir: Path
    fold_number: int
    task_id: int


def post_process_jobs(jobs: list[ParallelJobDict]) -> None:
    """Post process the jobs after they have been submitted."""
    for one_job in jobs:
        job_object = one_job["job"]
        fold_number = one_job["fold_number"]
        task_id = one_job["task_id"]
        split_number = one_job.get("split_number", None)
        custom_study = one_job["custom_study"]
        config_id = one_job.get("config_id", None)
        print(
            f"Waiting for job {job_object.job_id} with task {task_id} on fold "
            f"{fold_number} to finish with split {split_number} and config {config_id} "
            f"on study {custom_study}"
        )
        try:
            result_config = job_object.result()
        except Exception as e:
            print(f"Job {job_object.job_id} failed with exception {e}")
            continue
        print(f"\n\n\n\n\nJob {job_object.job_id} finished with result: {result_config}\n\n\n\n\n")


ARGUMENTS = {
    "--run_config": dict(type=str, help="Path to run config file"),
    "--exp_dir": dict(type=Path, help="Path to experiment directory", default=Path("experiments")),
    "--seed": dict(type=int, help="Seed for experiment", default=42),
    "--manipulators": dict(type=str, nargs="+", help="Manipulators to use", default=[]),
    "--model_name": dict(type=str, help="Model to use", default="reg_cocktails"),
    "--metric": dict(type=str, help="Metric to use", default="accuracy"),
    "--data_loader": dict(type=str, help="Data loader to use", default="openml"),
    "--exp_name": dict(type=str, help="Experiment name to use", default="Test"),
    "--device": dict(type=str, help="Device to use", default="cpu"),
    "--additional_metrics": dict(
        type=str,
        nargs="+",
        help="Additional metrics to use",
        default=[
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "log_loss",
            "average_precision",
            "brier_score",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "jaccard",
            "jaccard_micro",
            "jaccard_macro",
            "jaccard_weighted",
            "jaccard_samples",
            "roc_auc_ovr",
            "zero_one_loss",
            "precision",
            "recall",
        ],
    ),
    "--store_preds": dict(action="store_true", help="Store predictions"),
    "--study": dict(type=int, help="OpenML Suite id"),
    "--custom_study": dict(type=str, help="Name of custom study"),
    "--overwrite": dict(action="store_true", help="Overwrite existing results"),
    "--partition": dict(type=str, help="Partition to use", default="bosch_cpu-cascadelake"),
    "--n_parallel_jobs": dict(type=int, help="Max number of parallel jobs", default=20),
    "--slurm_job_time_secs": dict(type=float, help="Total job time in seconds", default=3600),
    "--n_workers": dict(type=int, help="Number of workers", default=4),
    "--n_trials": dict(type=int, help="Number of trials", default=10),
    "--n_folds": dict(type=int, help="Number of folds", default=5),
    "--task_ids": dict(type=int, nargs="+", help="Task ids to use", default=[3]),
    "--n_splits": dict(type=int, help="Number of splits", default=5),
    "--n_configs": dict(type=int, help="Number of configs", default=1),
    "--split_id": dict(type=int, help="Split id", default=1),
    "--fold_number": dict(type=int, help="Fold number", default=0),
    "--config_id": dict(type=int, help="Config id", default=1),
    "--split_number": dict(type=int, help="Split number", default=0),
    "--task_id": dict(type=int, help="Task id", default=3),
    "--memory": dict(type=int, help="Memory to use (in GB)", default=6),
}

COMMON_ARGUMENTS = [
    "--run_config",
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
]

CLUSTER_ARGUMENTS = [
    "--partition",
    "--n_parallel_jobs",
    "--slurm_job_time_secs",
]

SEARCH_ARGUMENTS = [
    "--n_workers",
    "--n_trials",
]


def add_args_to_parser(
    parser: argparse.ArgumentParser,
    list_args: list[str, dict[str, Any]],
) -> argparse.ArgumentParser:
    """Add arguments to the parser."""
    print(f"Adding arguments: {list_args}")
    for arg in list_args:
        kwargs = ARGUMENTS[arg]
        parser.add_argument(arg, **kwargs)
    return parser


def add_common_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments common for all scripts to the parser."""
    add_args_to_parser(parser, COMMON_ARGUMENTS)
    return parser


def add_cluster_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for cluster specific scripts to the parser."""
    add_args_to_parser(parser, CLUSTER_ARGUMENTS)
    return parser


def add_search_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments for search specific scripts to the parser."""
    add_args_to_parser(parser, SEARCH_ARGUMENTS)
    return parser
