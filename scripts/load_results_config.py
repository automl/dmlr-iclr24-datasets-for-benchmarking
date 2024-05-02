"""Given an experiment dir, it creates a Results instance with all the result configs stored in the experiment dir."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Type

import joblib
import numpy as np
import pandas as pd

from tabular_data_experiments.results.result import ConfigSplitResults, Results
from tabular_data_experiments.utils.utils import add_args_to_parser, get_executer

parser = argparse.ArgumentParser(description="Load results config from a given experiment dir")
parser.add_argument("--exp_dir", type=Path, help="Experiment dir", required=True)
parser.add_argument("--load_new", action="store_true", help="Load new results")
parser.add_argument("--config_split", action="store_true", help="Load results of the config split format")
parser.add_argument("--name", type=str, help="File name of the result file", default="results")
parser.add_argument("--slurm", action="store_true", help="Run on slurm")
parser = add_args_to_parser(parser,
                   ["--partition", "--n_workers"])
parser.add_argument("--max_folds", type=int, help="Max folds to load", default=4)
parser.add_argument("--max_splits", type=int, help="Max splits to load", default=5)
args = parser.parse_args()


def check_if_result_in_df(
    df: pd.DataFrame,
    file: Path,
    max_folds: int = 4,
    max_splits: int = 5
) -> bool:
    """Check if the result in the file is already in the df."""
    file_parts = file.parts
    # method is 6th last element
    method = file_parts[-6]
    # task is 5th last element
    task = int([part for part in file_parts if "task" in part][0].split("_")[1])
    # fold is 4th last element
    fold = int([part for part in file_parts if "fold" in part][0].split("_")[1])
    print(f"{file=} {method=} {task=} {fold=}")
    if int(fold) >= max_folds:
        print(f"Skipping {file} as it is fold {fold} and max folds is {max_folds}")
        return True
    # split is 3rd last element
    split = int([part for part in file_parts if "split" in part][0].split("_")[1])
    if int(split) >= max_splits:
        print(f"Skipping {file} as it is split {split} and max splits is {args.max_splits}")
        return True
    # config is 2nd last element
    config = int(file.parts[-2].split("_")[1])
    if df is None:
        return False
    # check if this element is nan
    row_idx = (method, fold, split, config)
    col_idx = ("test_roc_auc", task)
    # check if row_idx is in df index and col_idx is in df columns and
    # check if this element is nan
    if (
        (row_idx in df.index and col_idx in df.columns)
        and not np.isnan(df.loc[row_idx, col_idx])
        ):
        print(f"Skipping {file} as it is already in the df with value {row_idx=} {col_idx=}: {df.loc[row_idx, col_idx]}")
        return True

    return False


def load_json_close_file(
        file: Path, 
    df: pd.DataFrame = None,
    max_folds: int = 4,
    max_splits: int = 5
) -> dict[str, Any]:
    """Load a json file and close it."""
    # check if file's result is alreay in df
    absolute_path = file.resolve()
    print(f"Loading {file}")
    if (
        "default" not in absolute_path.as_posix()
        and check_if_result_in_df(df, file, max_folds=max_folds, max_splits=max_splits)
    ):
        print(f"Skipping {file} as it is already in the df")
        return None 
    with file.open("r") as f:
        return json.load(f)

def load_results_config(
    exp_dir: Path,
    result_class: Type[Results] = Results,
    n_jobs=4,
    config_split: bool = False,
    max_folds: int = 4,
    max_splits: int = 5
) -> Results:
    """Given an experiment dir, it creates a Results instance with all the result configs
    stored in the experiment dir."""
    if not exp_dir.exists():
        raise ValueError(f"Experiment dir {exp_dir} does not exist")
    df = None
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(load_json_close_file)(
            result_file, df=df if config_split else None, max_folds=max_folds)
        for result_file in exp_dir.glob("**/result.json")
    )
    results = [result for result in results if result is not None]
    print(f"Loaded {len(results)} results")
    return result_class.from_list(results, df=df)


if __name__ == "__main__":
    results_csv_path = args.exp_dir / f"{args.name}.csv"
    results_class = ConfigSplitResults if args.config_split else Results
    if results_csv_path.exists() and not args.load_new:
        print("Results already exist")
        results = results_class.from_csv(results_csv_path)
    else:
        log_dir = args.exp_dir / "gather_logs"
        if not log_dir.exists():
            log_dir.mkdir(parents=True)
        func_args = dict(
            exp_dir=args.exp_dir,
            result_class=results_class,
            n_jobs=args.n_workers,
            config_split=args.config_split,
            max_folds=args.max_folds,
            max_splits=args.max_splits
        )
        if args.slurm:
            slurm_executer = get_executer(
                partition=args.partition,
                cpu_nodes=args.n_workers,
                log_folder=log_dir,
                mem_per_cpu=2000,
            )
            results = slurm_executer.submit(
                load_results_config, **func_args)
            results = results.result()
        else:
            results = load_results_config(**func_args)
        results.df.to_csv(results_csv_path)
        # store time of creation
        (args.exp_dir / "results_creation_time.txt").write_text(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    # get average test accuracy table
    
    print(results.df)
