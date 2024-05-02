"""deprecated"""
import argparse
import json
from pathlib import Path
import shutil
import openml
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from tabular_data_experiments.data_loaders.openml import OpenMLLoader

from tabular_data_experiments.results.cd_diagram import LABEL_NAMES
from tabular_data_experiments.run_result_config import DataclassEncoder
from tabular_data_experiments.utils.data_utils import CUSTOM_SUITES, get_required_dataset_info, split_data

argparser = argparse.ArgumentParser()
argparser.add_argument("--n_splits", type=int, help="Number of splits for the dataset to generate", default=5)
argparser.add_argument("--output_dir", type=Path, help="Output directory", default=Path("splits"))
argparser.add_argument("--seed", type=int, help="Seed", default=42)
argparser.add_argument("--folds", type=int, help="Number of folds", default=4)
argparser.add_argument("--study", type=int, help="OpenML Suite id")
argparser.add_argument("--custom_study", type=str, help="Name of custom study")
argparser.add_argument("--task_ids", type=int, nargs="+", help="Task ids to run", default=[361481])
argparser.add_argument("--splitter", type=str, help="Splitter to use", default="stratified")
argparser.add_argument("--build_parquet", action="store_true", help="Build parquet files")
args = argparser.parse_args()



if __name__ == "__main__":
    if args.splitter == "stratified":
        cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    else: 
        raise NotImplementedError(f"Splitter {args.splitter} not implemented")

    task_ids = args.task_ids
    if args.study is not None:
        assert args.custom_study is None, "Cannot specify both study and custom_study"
        study = openml.study.get_suite(args.study)
        task_ids = study.tasks

    if args.custom_study is not None:
        assert args.study is None, "Cannot specify both study and custom_study"
        task_ids = CUSTOM_SUITES[args.custom_study]

    # continue_generation = True 
    # argument_file = args.output_dir / "arguments.json"
    # if args.output_dir.exists() and argument_file.exists():
    #     arguments = json.load(argument_file.open("r"))
    #     if arguments == vars(args):
    #         continue_generation = True

    for task in task_ids:
        task_dir = args.output_dir / str(task)
        pq_task_dir = Path("splits_pq") / str(task)
        task_dir.mkdir(parents=True, exist_ok=True)
        if pq_task_dir.exists() and (pq_task_dir / "splits.pq").exists():
            df = pd.read_parquet(pq_task_dir / "splits.pq")
            for fold in range(args.folds):
                fold_dir = task_dir / str(fold)
                if fold_dir.exists() and (fold_dir / "splits.json").exists():
                    print(f"Skipping task {task} on fold {fold} as it already exists")
                    continue
                else:
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    print(f"Generating splits for task {task} on fold {fold}")
                splits_json_dict = {}
                for i in range(args.n_splits):
                    splits_json_dict[i] = {
                            "train": df[df[f'openml_fold_{fold}'] != i & ~df[f'openml_fold_{fold}'].isna()].index.tolist(),
                            "val": df[df[f'openml_fold_{fold}'] == i].index.tolist(),
                            "test": df[df[f'openml_fold_{fold}'].isna()].index.tolist(),
                        }
                    json.dump(splits_json_dict, (fold_dir / "splits.json").open("w"), indent=4)

    arguments = vars(args)
    json.dump(arguments, (args.output_dir / "arguments.json").open("w"), indent=4, cls=DataclassEncoder)

