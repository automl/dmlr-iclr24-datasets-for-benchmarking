""" Utils for results """
from __future__ import annotations

from typing import Tuple

from pathlib import Path
from re import sub

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray
from sklearn.utils import shuffle

from tabular_data_experiments.results.constants import (
    dataset_collections,
    task_id_to_dataset_ids,
)
from tabular_data_experiments.results.result import ConfigSplitResults, Results


def get_incumbent_config_id(
    results: ConfigSplitResults, task_id: str, method: str, metric: str = "valid_roc_auc"
) -> NDArray[tuple[int, int]]:
    """get the config id of the incumbent configuration for a given task_id on all folds"""
    assert results.df is not None, "Results df is None"
    # select results for current task_id
    results = results.at(dataset=str(task_id), method=method)

    # mean across splits
    val_config_df = results.df.groupby(["fold", "config"]).mean()[metric]
    # get max config_id for each method and fold
    max_config_ids = val_config_df.groupby(["fold"]).idxmax()

    # remove nans
    max_config_ids = max_config_ids[~pd.isna(max_config_ids)]
    return max_config_ids.ravel()


def get_dataset_with_no_nans(r: Results) -> list[str]:
    """Get datasets which have no nans in the results"""
    df = r.df.copy()

    level = "task_id" if "task_id" in df.columns.names else "dataset"
    dataset_nan_info = df.T.groupby([level]).mean().isna().T.any()
    datasets_with_no_na = dataset_nan_info[dataset_nan_info == False].index.tolist()
    return datasets_with_no_na


def make_incumbent_trajectory(df: pd.DataFrame, opt_metric: str, plot_metric: str) -> pd.DataFrame:
    """
    Make the incumbent trajectory for a given dataframe. This is done by
    taking the best_so_far for the opt_metric and then filling the plot_metric
    with the value of the opt_metric if the opt_metric is the best_so_far and
    nan otherwise.

    Args:
        df(pd.DataFrame): dataframe
        opt_metric(str): metric to use for the best_so_far.
        plot_metric(str): other metric to be used for plotting

    Returns:
        df(pd.DataFrame): dataframe with incumbent trajectory
    """

    df["best_so_far"] = df[opt_metric].cummax()

    def make_nan(x):
        if x[opt_metric] == x["best_so_far"]:
            x[plot_metric] = x[plot_metric]
        else:
            x[plot_metric] = np.nan
        return x

    df = df.apply(make_nan, axis=1)
    df[plot_metric] = df[plot_metric].fillna(method="ffill")
    return df


def random_shuffle(
    df: pd.DataFrame,
    opt_metric: str = "valid_roc_auc",
    plot_metric: str = "test_roc_auc",
    n_shuffles: int = 15,
) -> pd.DataFrame:
    """
    Randomly shuffle the non-default configurations and then calculate the
    incumbent trajectory for each shuffle.

    Args:
        df(pd.DataFrame): dataframe
        opt_metric(str): metric to use for the best_so_far.
        plot_metric(str): other metric to be used for plotting
        n_shuffles(int): number of shuffles to perform

    Returns:
        df(pd.DataFrame): dataframe with incumbent trajectory
    """

    non_default_mask = np.array(df["config"] != "1")
    default_row = df[~non_default_mask]
    non_default_df = df[non_default_mask]
    shuffled_rows = []
    for _ in range(n_shuffles):
        shuffled_df = shuffle(non_default_df)
        with_default_df = pd.concat([default_row, shuffled_df])
        shuffled_df = make_incumbent_trajectory(with_default_df, opt_metric, plot_metric)
        shuffled_df["step"] = np.arange(1, shuffled_df.shape[0] + 1)
        shuffled_rows.append(shuffled_df)
    combined_df = pd.concat(shuffled_rows)age.to_numpy()to_list
    mean = combined_df.groupby("step").mean()[plot_metric].to_list()
    min = combined_df.groupby("step").min()[plot_metric].to_list()
    max = combined_df.groupby("step").max()[plot_metric].to_list()
    return mean, min, max


def camel_case(c) -> str:
    """
    Convert a string to camel case
    Args:
        c(str): string to convert
    Returns:
        camel_case(str): camel case string
    """
    return sub(r"(_|-)+", " ", c).title().replace(" ", "")


def get_citation_style(c: str) -> str:
    """
    Get the citation style for a given collection. This is done by
    splitting the collection name by "-" and then taking the first
    part and the last part. The first part is then converted to camel
    case and the last part is converted to the last two digits of the
    year.

    Args:
        c(str): collection

    Returns:
        citation(str): citation styled collection
    """
    splitted = c.split("-")
    name = splitted[0]
    conf_year = splitted[-1]
    year = conf_year[-3:-1]
    return f"{camel_case(name)} et al. {year}"


def store_table_for_metric(
    metric: str,
    collection: str,
    incumbent_results: Results,
    current_dataset_collections: dict[str, str],
    results_dir: Path,
) -> None:
    """
    Store the table for a given metric and collection

    Args:
        metric: metric to store
        collection: collection to store
        incumbent_results: incumbent results
        current_dataset_collections: current dataset collections
        results_dir: results dir

    Returns:
        None
    """
    print(f"Creating table for {metric}, {collection}")
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    incumbent_results.at(dataset=current_dataset_collections[collection], metric=f"test_{metric}").df.to_csv(
        tables_dir / f"test_{metric}_{collection}_comparison.csv"
    )


def get_order_of_labels_correlation(
    rank_results: pd.DataFrame,
):
    """
    Get the order of labels for the rank results. This is done by
    calculating the correlation between the rank results and the previous
    collection. The collection with the highest correlation is chosen as
    the next collection.

    Args:
        rank_results(pd.DataFrame): rank results

    Returns:
        order(list[str]): order of labels
    """
    rank_correlations = rank_results.corr(method="kendall")
    mse_correlations = rank_results.corr(method=lambda x, y: np.mean((x - y) ** 2))
    order = ["all_datasets"]
    for collection in rank_results.columns:
        if collection == "all_datasets":
            continue
        mse_collection_correlations = -1 * mse_correlations.loc[order[-1]].copy()
        tau_collection_correlations = rank_correlations.loc[order[-1]].copy()
        correlations_for_prev = pd.concat(
            [mse_collection_correlations, tau_collection_correlations], axis=1
        )  # .sort_values(ascending=False)
        correlations_for_prev.columns = ["mse", "tau"]
        correlations_for_prev = correlations_for_prev.sort_values(by=["mse", "tau"], ascending=False)
        # remove collections that are already in the order
        if all(o in correlations_for_prev.index for o in order):
            correlations_for_prev = correlations_for_prev.drop(order, axis=0)
        order.append(correlations_for_prev.index[0])
    return order


def get_median_datasets_in_collection(
    collection: str,
) -> float:
    """
    Get the median number of datasets in a collection

    Args:
        collection(str): collection

    Returns:
        median(float): median number of datasets in collection
    """
    path = Path(__file__).parent.parent.parent / "dataset_descripts/dataset_collection_final_datasets.csv"
    dataset_description = pd.read_csv(path, index_col=1)
    datasets = dataset_collections[collection]
    dataset_ids = [task_id_to_dataset_ids[dataset] for dataset in datasets]
    dataset_sizes = dataset_description.loc[dataset_ids, "Number of examples"]
    return dataset_sizes.median()


def get_order_of_labels_median_collection_size(
    rank_results: pd.DataFrame,
) -> list[str]:
    """
    Get the order of labels for the rank results. This is done by
    calculating the median number of datasets in a collection. The collection
    with the lowest median number of datasets is chosen as the next collection.

    Args:
        rank_results(pd.DataFrame): rank results

    Returns:
        order(list[str]): order of labels
    """

    # order labels according to median dataset size
    collections_to_be_sorted = {}
    for collection in rank_results.columns:
        if collection == "all_datasets":
            continue
        median_dataset_size = get_median_datasets_in_collection(collection)
        collections_to_be_sorted[collection] = median_dataset_size
    order = ["all_datasets"] + sorted(collections_to_be_sorted, key=collections_to_be_sorted.get, reverse=False)
    return order


def get_failed_dataset_info(
    current_results: Results,
    current_collections: dict[str, list[str]],
    file_path: Path | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    """
    Get the number of failed datasets for each method and collection

    Args:
        current_results: current results
        current_collections: Collections of datasets to use
        file_path: file path to store failed datasets. If None, no file is stored

    Returns:
        _description_
    """
    # Check per collection how many datasets are missing for ech method
    collection_info: dict[str, dict[str, int]] = {}
    failed_datasets: dict[str, list[str]] = {}
    for collection in current_collections:
        collection_info[collection] = {}
        datasets = current_collections[collection]
        collection_info[collection]["n_datasets"] = len(datasets)
        dataset_ids = [task_id_to_dataset_ids[dataset] for dataset in datasets]
        print(f"Collection: {collection}")
        try:
            for method in current_results.methods:
                if method not in failed_datasets:
                    failed_datasets[method] = []
                test_roc_df = (
                    current_results.at(method=[method], dataset_id=dataset_ids)
                    .df.groupby("method")
                    .mean()["test_roc_auc"]
                )
                datasets = test_roc_df.isna().any()[test_roc_df.isna().any()].index.tolist()
                collection_info[collection][method] = len(datasets)
                failed_datasets[method].extend(datasets)
        except Exception as e:
            print(f"Failed for {collection} with {e}")
    if file_path is not None:
        file_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(collection_info).to_csv(file_path / "failed_datasets.csv")

    return collection_info, failed_datasets


def get_collection_metadata(
    collection_datasets: dict,
) -> pd.DataFrame:
    """
    describe each collection, by telling number of datasets, number of numerical only datasets,
    number of datasets with cat, median number of missing values, median dataset size, QR3 for dataset size,
    median number of features, QR3 for median number of features. Raw numbers are available via the
    openml api
    """
    collection_meta = {}
    for collection in collection_datasets:
        task_ids = collection_datasets[collection]
        task_ids = [int(task_id) for task_id in task_ids]
        tasks = openml.tasks.get_tasks(task_ids, download_data=False)
        datasets = openml.datasets.get_datasets([task.dataset_id for task in tasks], download_data=False)
        qualities = {}
        num_numerical_datasets = 0
        num_cateforical_datasets = 0
        for dataset in datasets:
            qualities[dataset.name] = dataset.qualities
            if dataset.qualities["NumberOfSymbolicFeatures"] <= 1:
                num_numerical_datasets += 1
            else:
                num_cateforical_datasets += 1
        df = pd.DataFrame(qualities).T
        number_of_datasets = len(df)
        required_attributes = [
            "NumberOfNumericFeatures",
            "NumberOfSymbolicFeatures",
            "NumberOfFeatures",
            "NumberOfClasses",
            "NumberOfMissingValues",
            "NumberOfInstances",
            "MajorityClassPercentage",
            "MinorityClassPercentage",
        ]
        for column in required_attributes:
            df[column] = df[column].astype(float)

        median_df = df[required_attributes].median()
        median_df.index = ["median_" + attribute.replace("NumberOf", "n_") for attribute in required_attributes]
        q3_df = df[required_attributes].quantile(0.75)
        q3_df.index = ["q3_" + attribute.replace("NumberOf", "n_") for attribute in required_attributes]
        q1_df = df[required_attributes].quantile(0.25)
        q1_df.index = ["q1_" + attribute.replace("NumberOf", "n_") for attribute in required_attributes]
        # arrange in series
        combined = pd.concat([median_df, q3_df, q1_df])
        combined["number_of_datasets"] = number_of_datasets
        combined["number_of_numerical_datasets"] = num_numerical_datasets
        combined["number_of_categorical_datasets"] = num_cateforical_datasets
        collection_meta[collection] = combined

    collection_meta_df = pd.DataFrame(collection_meta).T
    return collection_meta_df


def load_results(
    model_results_dir: Path,
    methods: list,
    result_file_name: str = "results.csv",
    datasets: list[str] | None = None,
) -> ConfigSplitResults:
    """
    Load results from a directory and return a ConfigSplitResults object.

    Args:
        model_results_dir: Directory containing results.csv
        methods: List of methods to load
        result_file_name: Name of the result file. Default: results.csv

    Returns:
        ConfigSplitResults object
    """
    results = ConfigSplitResults()

    for method in methods:
        results = results.add_method(
            ConfigSplitResults.from_csv(
                model_results_dir / method / result_file_name,
            )
        )
    if datasets is None:
        datasets = dataset_collections["all_datasets"]

    results = results.at(dataset=datasets)

    return results


# Methods to compute rank


def ranks_random_sets(
    df: pd.DataFrame, tasks: list, start_with_size: int = 3, n_rep: int = 50, seed: int = 1
) -> pd.DataFrame:
    """
    For n repetitions do:
        shuffle lists of datasets
        compute average rank across increasing sets of did; add did one by one
    Return fancy multi-index datafram
    """
    rng = np.random.RandomState(seed)

    # Compute rank for all dids
    ranks = df.rank(ascending=False, axis=0)

    # pre-defined list of outer colnames
    colnames_outer = [f"rep_{r}" for r in range(n_rep)]

    all_res = []
    for r in range(n_rep):
        rank = []
        idx = rng.permutation(tasks)
        for i, _ in enumerate(idx):
            rs = ranks[idx[:i]].mean(axis=1)
            rank.append(rs)
        rank = pd.concat(rank, axis=1)
        rank.rename(columns=lambda x: f"size_{x}", inplace=True)
        all_res.append(rank)

    all_res = pd.concat(all_res, axis=1)
    all_res.columns = pd.MultiIndex.from_product([colnames_outer, all_res.columns.unique()], names=["rep", "size"])
    return all_res


def ranks_by_frequency(df: pd.DataFrame, tasks: list, task_dc: dict[str, list[str]]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute rank for all dids, based on frequency of use in suites
    """

    # This is a dc with all suites and the used datasets, taken from MM on June 6th
    # TODO: Check whether this is up-to-date

    # Create matrix with suites x dids; set to 1 if dataset is in suite
    task_df = pd.DataFrame(index=task_dc.keys(), columns=tasks).fillna(0)
    for suite in task_dc:
        # only use tasks which are in the tasks list
        suite_tasks = [int(task) for task in task_dc[suite] if task in tasks]
        task_df.loc[suite, suite_tasks] = 1

    # Sum columns (= how often is did used)
    # Sort in descending order (= most frequently used did is at top)
    # Get list of dids
    tasks_by_frequency = task_df.sum(axis=0).sort_values(ascending=False).index
    # Compute rank across set of did; increase set in each iteration by one did
    ranks = []
    for i, _ in enumerate(tasks_by_frequency):
        r = df[tasks_by_frequency[:i]].rank(ascending=False).mean(axis=1)
        ranks.append(r)
    ranks = pd.concat(ranks, axis=1)
    # convert tasks to int
    ranks.columns = ranks.columns.astype(int)

    return ranks, task_df.sum(axis=0)


def get_results_after_cutoff(
    results: ConfigSplitResults,
    cutoff: float = 3600,
) -> ConfigSplitResults:
    """
    Sets a walltime cutoff limit and returns the results that are below the cutoff. Others are set to nan

    Args:
        results(ConfigSplitResults): Results to be filtered
        cutoff(float): Cutoff in seconds

    Returns:
        ConfigSplitResults: Filtered results
    """
    return ConfigSplitResults(results.df[results.df["total_walltime"] < cutoff].copy())


def get_total_time_per_method(
    current_results: ConfigSplitResults,
    n_configs: int = 100,
    n_folds: int = 4,
    n_splits: int = 5,
):
    """
    Gets dataframe with total time per method
    """
    all_pos_df = current_results.df[
        current_results.df["total_walltime"] > 0 & (current_results.df["total_walltime"] < 86400)
    ].copy()
    walltime_df = all_pos_df.groupby(["method"]).sum()["total_walltime"]
    walltime_df = walltime_df[walltime_df > 0].fillna(0)
    return (walltime_df * n_splits).sum(axis=1) / 86400
