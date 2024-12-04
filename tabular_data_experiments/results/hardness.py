"""
This module contains functions to calculate and store the hardness of datasets
based on different metrics.

The calculate_hardness_grin function calculates the hardness for each dataset based
on grin, which is calculated using the following formula: 
hardness = min(score_hgbt, score_resnet) - max(score_logreg, score_dt), which is later 
min max normalized.

The calculate_hardness_std function calculates the hardness for
each dataset based on std between the performances of a model, which is min max
normalized.

The store_data function stores the hardness data to file.

The load_data_for_methods function loads the data for each method.

The get_age function loads the year info for each dataset.

The get_hardness_age function merges the hardness and age dataframes.
"""
from __future__ import annotations

from pathlib import Path

import openml
import pandas as pd

from tabular_data_experiments.results.constants import (
    MissingResultsHandler,
    dataset_collections,
)
from tabular_data_experiments.results.result import Results


def calculate_hardness_grin(
    combined_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates the hardness for each dataset based on grin
    which is calculated using the following formula:
    hardness = min(score_hgbt, score_resnet) - max(score_logreg, score_dt)
    which is later min max normalized.

    Args:
        results(dict[str, pd.DataFrame]): results for each method

    Returns:
        pd.DataFrame: hardness for each dataset
    """
    max_per_data = combined_results.max(axis=1)
    min_per_data = combined_results.min(axis=1)
    min_per_data[min_per_data > 0.5] = 0.5
    diff = max_per_data - min_per_data
    min_scores = combined_results[["hgbt", "resnet"]].min(axis=1)
    max_scores = combined_results[["tree", "linear"]].max(axis=1)
    min_scores = (min_scores - min_per_data) / diff
    max_scores = (max_scores - min_per_data) / diff
    hardness = min_scores - max_scores
    combined_results["hardness"] = hardness
    return combined_results


def calculate_hardness_std(results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the hardness for each dataset based on std
    which is min max normalized

    Args:
        results(dict[str, pd.DataFrame]): results for each method

    Returns:
        pd.DataFrame: hardness for each dataset
    """
    results = results.copy()
    max_per_data = results.max(axis=1)
    min_per_data = results.min(axis=1)
    min_per_data[min_per_data > 0.5] = 0.5
    diff = max_per_data - min_per_data
    results = ((results.transpose() - min_per_data) / diff).transpose()
    std_full_data = results.std(axis=1)
    # normalize std
    std_full_data = (std_full_data - std_full_data.min()) / (std_full_data.max() - std_full_data.min())
    results["hardness"] = std_full_data
    return results


def store_data(
    exp_dir: Path,
    std_full_data: pd.DataFrame,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Stores the hardness data to file

    Args:
        exp_dir(Path): experiment directory
        std_full_data(pd.DataFrame): hardness data
        suffix(str): suffix for the file name

    Returns:
        pd.DataFrame: hardness data
    """
    std_full_data = std_full_data.sort_values(by="hardness", ascending=False)
    std_full_data = std_full_data.reset_index()
    std_full_data["dataset_id"] = std_full_data["task_id"].apply(
        lambda x: openml.tasks.get_task(int(x), download_data=False).dataset_id
    )
    # std_full_data = std_full_data[["dataset_id", "hardness", "task_id"]]
    std_full_data = std_full_data.dropna()
    std_full_data.to_csv(exp_dir / f"hardness_{suffix}.csv")
    return std_full_data


def load_data_for_methods(
    exp_dir: Path,
    criteria: str,
    metric: str,
    methods: list[str],
    missing_results_handler: MissingResultsHandler = MissingResultsHandler.impute_nan,
) -> pd.DataFrame:
    """
    Loads the data for each method

    Args:
        args: arguments
        methods: methods to load data for

    Returns:
        dict[str, pd.DataFrame]: results for each method
    """
    results = {}
    for method in methods:

        # TODO: read from default results also for std
        r = Results.from_csv(exp_dir / method / "results.csv")
        r.at(dataset=dataset_collections["all_datasets"])
        # handle missing results
        r.handle_missing_results(missing_value_handler=missing_results_handler)
        results[method] = (
            r.df[f"test_{metric}"].groupby("method").mean().reset_index().drop("method", axis=1).dropna(axis=1)
        )

    combined_results = pd.concat(results).dropna(axis=1).droplevel(1).T
    return combined_results


def get_age(path_to_year_info: str) -> pd.DataFrame:
    """
    Loads the year info for each dataset

    Args:
        path_to_year_info(str): path to year info file

    Returns:
        pd.DataFrame: year info for each dataset
    """

    year_info = pd.read_csv(path_to_year_info)

    age = year_info[["Dataset ID", "Year"]]
    age = age.rename(columns={"Dataset ID": "dataset_id", "Year": "age"})
    age["age"] = pd.to_datetime(age["age"], format="%Y", errors="coerce")
    age["age"] = age["age"].apply(lambda x: x.year)
    age = age.sort_values(by="age")
    age = age.dropna()
    age["age"] = age["age"].astype(int)
    return age


def get_hardness_age(hardness, age):
    hardness = hardness.merge(age, on="dataset_id")
    hardness = hardness.dropna()
    hardness = hardness.sort_values(by="age")
    hardness = hardness.dropna()
    return hardness
