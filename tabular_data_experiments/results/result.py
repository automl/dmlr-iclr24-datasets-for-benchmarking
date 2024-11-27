"""This module contains the Results class which is used to store the results of a run"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import openml
import pandas as pd

from tabular_data_experiments.results.constants import (
    MissingResultsHandler,
    task_id_to_dataset_ids,
)


@dataclass
class Results:
    # Big ass predefined dictionary
    df: pd.DataFrame | None = None
    index: ClassVar[tuple[str, ...]] = ("method", "fold")
    columns: ClassVar[tuple[str, ...]] = ("metric", "task_id")

    @classmethod
    def from_list(self, results: list[dict[str, Any]], *, df: pd.DataFrame | None, dropna: bool = True) -> "Results":
        methods = list()
        metrics = list()
        task_ids = list()
        folds = list()
        for result in results:
            methods.append(result["run_config"]["model_name"])
            metrics.extend(result["run_config"]["additional_metrics"])
            task_ids.append(result["run_config"]["task_id"])
            folds.append(result["run_config"]["fold_number"])

        # The unique, methods, times, metrics and splits present
        methods = np.unique(methods).astype(str)
        metrics = np.unique(metrics).astype(str)
        folds = np.unique(folds).astype(int)
        task_ids = np.unique(task_ids).astype(int)

        # make train, val, test metrics separate metrics
        data_splits = ["train", "valid", "test"]
        metrics = ["_".join(metric) for metric in product(data_splits, metrics)]
        # add time metrics
        metrics.extend(["average_split_fit_time", "average_split_predict_time", "total_walltime"])

        index = pd.MultiIndex.from_product(
            [methods, folds],
            names=self.index,
        )

        columns = pd.MultiIndex.from_product(
            [metrics, task_ids],
            names=self.columns,
        )

        df = pd.DataFrame(columns=columns, index=index)
        df.sort_index(inplace=True)

        for result in results:
            method = str(result["run_config"]["model_name"])
            fold = int(result["run_config"]["fold_number"])
            task_id = int(result["run_config"]["task_id"])
            for metric in result["run_config"]["additional_metrics"]:
                for split in data_splits:
                    split_metrics = result["result"][1][f"{split}_metrics"]
                    if split_metrics is None:
                        value = np.nan
                    else:
                        value = split_metrics.get(metric, np.nan)
                    df.loc[(method, fold), (f"{split}_{metric}", task_id)] = value
            for metric in ["average_split_fit_time", "average_split_predict_time", "total_walltime"]:
                value = result["result"][1][metric]
                df.loc[(method, fold), (metric, task_id)] = value
            if "num_params" in result["result"][1]["additional_info"]:
                df.loc[(method, fold), ("num_params", task_id)] = result["result"][1]["additional_info"]["num_params"]
        # Drop full NaN rows
        if dropna:
            df = df[df.any(axis=1)]
        return Results(df)

    def at(
        self,
        *,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ) -> Results:
        """Use this for slicing in to the dataframe to get what you need"""
        assert self.df is not None, "No results to slice"
        df = self.df
        items = {
            "method": method,
            "fold": fold,
        }
        df = self._filter_df(
            df,
            items,
            dataset=dataset,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            metric=metric,
        )

        return Results(df)

    def _filter_df(
        self,
        df: pd.DataFrame,
        index_items: dict[str, Any],
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ):
        df = self._search_index_from_items(df, index_items)

        if dataset_id:
            df = self._search_dataset_ids(dataset_id, df)

        if dataset_name:
            df = self._search_dataset_name(dataset_name, df)

        if dataset:
            df = self._search_datasets(dataset, df)

        if metric:
            df = self._search_metric(metric, df)
        if df.empty:
            raise RuntimeError("No results found")

        return df

    def _search_dataset_name(self, dataset_name, df):
        _dataset_name = dataset_name if isinstance(dataset_name, list) else [dataset_name]
        dataset = [self.dataset_name_to_dataset.get(dataset_name, None) for dataset_name in _dataset_name]
        dataset = [dataset_name for dataset_name in dataset if dataset_name is not None]
        df = self._search_datasets(dataset, df)
        return df

    def _search_dataset_ids(self, dataset_id, df):
        _dataset_ids = dataset_id if isinstance(dataset_id, list) else [dataset_id]
        dataset = [self.dataset_ids_to_dataset.get(dataset_id, None) for dataset_id in _dataset_ids]
        dataset = [dataset_id for dataset_id in dataset if dataset_id is not None]
        df = self._search_datasets(dataset, df)
        return df

    def _search_index_from_items(self, df, items):
        for name, item in items.items():
            if item is None:
                continue
            idx: list = item if isinstance(item, list) else [item]
            df = df[df.index.get_level_values(name).isin(idx)]
            if not isinstance(item, list):
                df = df.droplevel(name, axis="index")
        return df

    def _search_metric(self, metric, df):
        _metric = metric if isinstance(metric, list) else [metric]
        df = df.T.loc[df.T.index.get_level_values("metric").isin(_metric)].T
        if not isinstance(metric, list):
            df = df.droplevel("metric", axis="columns")
        return df

    def _search_datasets(self, dataset, df):
        _dataset = dataset if isinstance(dataset, list) else [dataset]
        level = "task_id" if "task_id" in df.columns.names else "dataset"  # backwards compatibility
        df = df.T.loc[df.T.index.get_level_values(level).isin(_dataset)].T
        if not isinstance(dataset, list):
            df = df.droplevel(level, axis="columns")
        return df

    @staticmethod
    def _get_dataset_id_from_openml_task(task_id) -> int | np.nan:
        return task_id_to_dataset_ids.get(task_id, np.nan)

    @staticmethod
    def _get_dataset_name_from_openml_task(task_id) -> str:
        task = openml.tasks.get_task(int(task_id), download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        return dataset.name

    @property
    def methods(self) -> list[str]:
        if self.df is None:
            return []
        return list(self.df.index.get_level_values("method").unique())

    @property
    def folds(self) -> list[int]:
        if self.df is None:
            return []
        return list(self.df.index.get_level_values("fold").unique())

    @property
    def task_ids(self) -> list[str]:
        if self.df is None:
            return []
        if "task_id" in self.df.columns.names:
            return list(self.df.columns.get_level_values("task_id").unique())
        else:  # for backwards compatibility
            return list(self.df.columns.get_level_values("dataset").unique())

    @property
    def metrics(self) -> list[str]:
        if self.df is None:
            return []
        return list(self.df.columns.get_level_values("metric").unique())

    @property
    def dataset_ids_to_dataset(self) -> dict[int, str]:
        return {self._get_dataset_id_from_openml_task(dataset): dataset for dataset in self.task_ids}

    @property
    def dataset_name_to_dataset(self) -> dict[str, str]:
        return {self._get_dataset_name_from_openml_task(dataset): dataset for dataset in self.task_ids}

    @property
    def dataset_names(self) -> list[str]:
        return list(self.dataset_name_to_dataset.keys())

    @property
    def dataset_ids(self) -> list[int]:
        return list(self.dataset_ids_to_dataset.keys())

    @classmethod
    def from_csv(cls, path: str | Path) -> "Results":
        """
        Load results from a csv file.

        Args:
            path: Path to csv file.

        Returns:
            ConfigSplitResults
        """
        df = pd.read_csv(
            path,
            index_col=list(range(len(cls.index))),
            header=list(range(len(cls.columns))),
            engine="c",
        )
        # df.index.names = cls.index
        return cls(df)

    def add_method(self, method_results: Results) -> "Results":
        """
        Add results from a method to the current results.

        Args:
            method_results: Results of the method.

        Returns:
            Results
        """
        if self.df is not None:
            df = self.df
            df = pd.concat([df, method_results.df], axis=0)
        else:
            df = method_results.df
        return self.__class__(df)

    def drop_method(self, method_name: list[str]) -> "Results":
        """
        Drop results from a method to the current results.

        Args:
            method_name: Name of the method.

        Returns:
            Results
        """
        assert self.df is not None, "No results to drop"
        df = self.df
        df = df.drop(method_name, axis=0)
        return self.__class__(df)

    def get_metric_table(
        self,
        *,
        metric: str,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
    ) -> pd.DataFrame:
        """Get a table of the metrics"""
        assert self.df is not None, "No results to slice"
        df = self.df.copy()
        df = self._filter_df(
            df,
            {"method": method, "fold": fold},
            dataset=dataset,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            metric=metric,
        )
        # take average over folds
        df = df.groupby(["method"]).mean()
        # get table where rows are datasets and columns are methods
        df = df.T

        return df

    def handle_missing_results(
        self,
        *,
        missing_value_handler: MissingResultsHandler = MissingResultsHandler.impute_nan,
        missing_value_metric: str = "train_accuracy",
    ) -> Results:
        """
        Handle nans in the results.


        Args:
            inplace: Whether to do the operation inplace.
            missing_value_handler: How to handle missing values.

        Returns:
            Results
        """
        assert self.df is not None, "No results to handle nans"
        df = self._filter_df(self.df, index_items={}, metric=missing_value_metric).copy()
        # nan info about the tasks
        dataset_nan_info = df.T.groupby(["task_id"]).mean().isna()
        # Drop tasks with nans for all method and folds
        not_all_nan_tasks = dataset_nan_info[~dataset_nan_info.all(axis=1)].index.tolist()
        all_nan_tasks = dataset_nan_info[dataset_nan_info.all(axis=1)].index.tolist()
        print(
            f"Dropping {len(all_nan_tasks)} tasks {all_nan_tasks} with all nans"
            f" and keeping {len(not_all_nan_tasks)} tasks"
        )
        df = self._filter_df(self.df, index_items={}, dataset=not_all_nan_tasks).copy()
        # if there are still nans, handle them
        if missing_value_handler == MissingResultsHandler.impute_nan:
            df = df.fillna(0)
        elif missing_value_handler == MissingResultsHandler.drop_nan:
            df = df.dropna()
        else:
            raise ValueError("Invalid missing value handler")
        return self.__class__(df)


@dataclass
class ConfigSplitResults(Results):
    index: ClassVar[tuple[str, ...]] = ("method", "fold", "split", "config")

    @classmethod
    def from_list(
        cls, results: list[dict[str, Any]], *, df: pd.DataFrame | None = None, dropna: bool = True
    ) -> "ConfigSplitResults":
        print("Getting unique methods, metrics, datasets, and splits")
        data_splits = ["train", "valid", "test"]

        if df is None:
            df = cls._create_df(results, data_splits)

        df.sort_index(inplace=True)
        df = df.astype("float16")  # Change the precision of the columns to float16

        print("Filling dataframe")
        for result in results:
            if result is None:
                continue
            method = str(result["run_config"]["model_name"])
            fold = int(result["run_config"]["fold_number"])
            split_id = int(result["run_config"]["split_id"])
            config_id = int(result["run_config"]["config_id"])
            task_id = int(result["run_config"]["task_id"])
            row_index = (method, fold, split_id, config_id)
            for metric in result["run_config"]["additional_metrics"]:
                for split in data_splits:
                    split_metrics = result["result"][1][f"{split}_metrics"]
                    if split_metrics is None:
                        value = np.nan
                    else:
                        value = split_metrics.get(metric, np.nan)
                    col_index = (f"{split}_{metric}", task_id)
                    df.loc[row_index, col_index] = value

            for metric in ["average_split_fit_time", "average_split_predict_time", "total_walltime"]:
                # Set times to 0 if errored out
                if "error" in result["result"][1]["additional_info"]:
                    value = -1 * 0.1 * 86400
                else:
                    value = result["result"][1][metric]
                col_index = (metric, task_id)
                df.loc[row_index, col_index] = value

            if "num_params" in result["result"][1]["additional_info"]:
                col_index = ("num_params", task_id)
                df.loc[row_index, col_index] = value
            if "best_iteration" in result["result"][1]["additional_info"]:
                col_index = ("best_iteration", task_id)
                df.loc[row_index, col_index] = value

            # mismatched name in CPU models
            if "best_iter" in result["result"][1]["additional_info"]:
                col_index = ("best_iteration", task_id)
                df.loc[row_index, col_index] = value

        print("Done filling dataframe")

        # Drop full NaN rows
        if dropna:
            df = df[df.any(axis=1)]
        # sort index
        return ConfigSplitResults(df)

    @classmethod
    def _create_df(
        cls,
        results: list[dict[str, Any]],
        data_splits: list[str],
    ):
        methods = [result["run_config"]["model_name"] for result in results]
        metrics = [metric for result in results for metric in result["run_config"]["additional_metrics"]]
        datasets = [result["run_config"]["task_id"] for result in results]
        folds = [result["run_config"]["fold_number"] for result in results]
        splits = [result["run_config"]["split_id"] for result in results]
        configs = [result["run_config"]["config_id"] for result in results]

        # The unique methods, times, metrics, and splits present
        methods = np.unique(methods).astype(str)
        metrics = np.unique(metrics).astype(str)
        folds = np.unique(folds).astype(int)
        datasets = np.unique(datasets).astype(int)
        splits = np.unique(splits).astype(int)
        configs = np.unique(configs).astype(int)

        # Make train, val, test metrics separate metrics
        metrics = ["_".join(metric) for metric in product(data_splits, metrics)]
        # Add time metrics
        metrics.extend(["average_split_fit_time", "average_split_predict_time", "total_walltime"])

        index = pd.MultiIndex.from_product(
            [methods, folds, splits, configs],
            names=cls.index,
        )

        columns = pd.MultiIndex.from_product(
            [metrics, datasets],
            names=cls.columns,
        )

        print("Creating empty dataframe")
        df = pd.DataFrame(columns=columns, index=index)
        return df

    def at(
        self,
        *,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        split: int | list[int] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
        config: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        metric: str | list[str] | None = None,
    ) -> Results:
        """Use this for slicing in to the dataframe to get what you need"""
        assert self.df is not None, "No results to slice"
        df = self.df
        items = {"method": method, "fold": fold, "split": split, "config": config}
        df = self._filter_df(
            df,
            items,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset=dataset,
            metric=metric,
        )

        return ConfigSplitResults(df)

    @property
    def splits(self) -> list[int]:
        if self.df is None:
            return []
        return list(self.df.index.get_level_values("split").unique())

    @property
    def configs(self) -> list[int]:
        if self.df is None:
            return []
        return list(self.df.index.get_level_values("config").unique())

    def get_incumbent_results(
        self,
        *,
        metric: str,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        split: int | list[int] | None = None,
        config: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
    ) -> Results:
        """
        Get the incumbent results for each method and fold. This is the config
        with the best validation score for each method and fold.
        """
        assert self.df is not None, "No results to slice"
        # filter df
        df = self.df.copy()
        df = self._filter_df(
            df,
            {"method": method, "fold": fold, "split": split, "config": config},
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset=dataset,
        )

        # only keep the max config_id for each method and fold
        config_df = df.groupby(["method", "fold", "config"]).mean()  # take mean across splits

        # select df for interested metric
        val_config_df = config_df[metric]
        # get max config_id for each method and fold
        max_config_ids = val_config_df.groupby(["method", "fold"]).idxmax()

        for task_id in self.task_ids:
            if task_id not in max_config_ids.columns:
                continue
            # get the max config_id for each method and fold
            method_to_max_config_id = max_config_ids.loc[:, task_id].values.ravel()
            # remove nans
            method_to_max_config_id = method_to_max_config_id[~pd.isna(method_to_max_config_id)]
            # only keep the max config_id for each method and fold
            config_df.loc[:, (slice(None), task_id)] = config_df.loc[method_to_max_config_id, (slice(None), task_id)]

        # config_df contains the incumbent config for each method and fold and nans for the other configs
        # so we can just take the max across the config, axis
        incumbent_results = Results(config_df.groupby(["method", "fold"]).max())
        return incumbent_results

    def get_metric_table(
        self,
        *,
        metric: str,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        split: int | list[int] | None = None,
        config: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
        data_split: Literal["train", "valid", "test"] = "test",
    ) -> pd.DataFrame:
        """Get a table of the metrics. This is the same as the parent class
        but we need to override it to use get_incumbent_results which returns
        a Results object containing the incumbent results for each method and
        fold calculated using the valid_metric. We then call get_metric_table on
        this object to get the data_split_metric table for the incumbent results.
        """
        assert self.df is not None, "No results to slice"
        incumbent_results = self.get_incumbent_results(
            metric=f"valid_{metric}",
            method=method,
            fold=fold,
            dataset=dataset,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            split=split,
            config=config,
        )
        return incumbent_results.get_metric_table(metric=f"{data_split}_{metric}")

    def get_default_results(
        self,
        *,
        metric: str | list[str] | None = None,
        method: str | list[str] | None = None,
        fold: int | list[int] | None = None,
        split: int | list[int] | None = None,
        dataset: str | list[str] | None = None,
        dataset_id: int | list[int] | None = None,
        dataset_name: str | list[str] | None = None,
    ) -> Results:
        """
        Get the default results for each method and fold.
        """
        assert self.df is not None, "No results to slice"
        # filter df
        df = self.df.copy()
        df = self._filter_df(
            df,
            {"method": method, "fold": fold, "split": split, "config": 1},  # only keep the default config
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset=dataset,
            metric=metric,
        )

        return Results(df.groupby(["method", "fold"]).mean())
