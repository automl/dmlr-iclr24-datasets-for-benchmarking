"""
for each dataset and each method, goes through the available results
and finds the best config id for each fold and each method and saves it to a file.
"""
import argparse

import json
from pathlib import Path
import openml
import pandas as pd

from tabular_data_experiments.results.result import ConfigSplitResults
from tabular_data_experiments.results.utils import get_incumbent_config_id
from tabular_data_experiments.utils.suites import CUSTOM_SUITES
from tabular_data_experiments.utils.utils import add_args_to_parser

parser = argparse.ArgumentParser(description="Run a experiment with a given config")
parser.add_argument("--exp_dir", type=Path, help="Experiment dir", required=True)
parser.add_argument("--name", type=str, help="File name of the result file", default="results")
parser.add_argument("--n_jobs", type=int, help="Number of jobs", default=4)
parser.add_argument("--out_dir", type=Path, help="Output dir", default="best_config_ids")
add_args_to_parser(parser, ["--task_ids", "--custom_study", "--study", "--n_folds",
                            "--model_name", "--metric"])


args = parser.parse_args()
if __name__ == "__main__":
    if not args.out_dir.exists():
        args.out_dir.mkdir(parents=True)

    task_ids = args.task_ids
    if args.study is not None:
        assert args.custom_study is None, "Cannot specify both study and custom_study"
        study = openml.study.get_suite(args.study)
        task_ids = study.tasks

    if args.custom_study is not None:
        assert args.study is None, "Cannot specify both study and custom_study"
        task_ids = CUSTOM_SUITES[args.custom_study]
 
    model_dir = args.exp_dir / args.model_name
    model_results = ConfigSplitResults.from_csv(
        model_dir / "results.csv"
    )
    incumbents = []
    for task_id in task_ids:
        incumbent_dict = {
            "task_id": task_id,
        }
        try:
            max_config_ids = get_incumbent_config_id(
                results=model_results,
                task_id=task_id,
                metric=f"valid_{args.metric}",
                method=args.model_name,
                )
        except KeyError:
            print(f"Could not find results for task {task_id} in {model_dir}")
            continue

        # drop nans
        max_config_ids = [x for x in max_config_ids if x[1] is not None]

        for fold, config_id in max_config_ids:
            incumbent_dict[f"fold_{fold}"] = int(config_id)
        incumbents.append(incumbent_dict)
    # Check if incumbents already exist
    # if (args.out_dir / f"{args.model_name}_incumbents.csv").exists():
    #     incumbents_df = pd.read_csv(
    #         args.out_dir / f"{args.model_name}_incumbents.csv"
    #     )
    #     incumbents_df = pd.concat([incumbents_df, pd.DataFrame(incumbents)], axis=0)
    #     # Remove duplicates task ids
    #     incumbents_df = incumbents_df.drop_duplicates(subset=["task_id"])
    #     incumbents_df.to_csv(
    #         args.out_dir / f"{args.model_name}_incumbents.csv",
    #         index=False
    #     )
    # else:
    pd.DataFrame(incumbents).to_csv(
        args.out_dir / f"{args.model_name}_incumbents.csv",
        index=False)


