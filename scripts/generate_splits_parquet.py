import argparse
from pathlib import Path
import numpy as np
import openml
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from tabular_data_experiments.utils.suites import CUSTOM_SUITES

argparser = argparse.ArgumentParser()
argparser.add_argument("--n_splits", type=int, help="Number of splits for the dataset to generate", default=5)
argparser.add_argument("--output_dir", type=Path, help="Output directory", default=Path("splits_pq"))
argparser.add_argument("--seed", type=int, help="Seed", default=42)
argparser.add_argument("--n_folds", type=int, help="Number of folds", default=4)
argparser.add_argument("--study", type=int, help="OpenML Suite id")
argparser.add_argument("--custom_study", type=str, help="Name of custom study")
argparser.add_argument("--task_ids", type=int, nargs="+", help="Task ids to run", default=[31])
argparser.add_argument("--splitter", type=str, help="Splitter to use", default="stratified")
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

    for task in task_ids:
        try:
            task_dir = args.output_dir / str(task)
            task_dir.mkdir(parents=True, exist_ok=True)
            if (task_dir / "splits.pq").exists():
                continue

            openml_task = openml.tasks.get_task(task)
            dataset = openml.datasets.get_dataset(openml_task.dataset_id)
            X, y, categorical_indicator, _ = dataset.get_data(
                dataset_format="dataframe",
                target=openml_task.target_name,
            )
            n_samples = int(dataset.qualities["NumberOfInstances"])
            task_info = np.empty((n_samples, args.n_folds))
            task_info[:] = np.nan
            task_pq_df = pd.DataFrame(task_info, columns=[f"openml_fold_{i}" for i in range(args.n_folds)])
            for i in range(args.n_folds):
                train_indices, _ = openml_task.get_train_test_split_indices(fold=i)
                y_train = y.loc[train_indices]
                cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
                for j, (train_idx, test_idx) in enumerate(cv.split(train_indices, y_train)):
                    test_idx_fold = train_indices[test_idx]
                    task_pq_df.loc[test_idx_fold, f"openml_fold_{i}"] = j

            task_pq_df.to_parquet(task_dir / "splits.pq")
        except Exception as e:
            print(f"Failed for task {task}")
            print(e)
