from __future__ import annotations

from typing import Any, Optional

from pathlib import Path

import colorcet as cc
import numpy as np
import pandas as pd
import seaborn as sns
from autorank import autorank, plot_stats
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

from tabular_data_experiments.results.constants import (
    CPU_MODELS,
    FINAL_LABEL_NAMES,
    FINAL_MODELS,
    GPU_MODELS,
    MODELS_TO_COLORS,
)
from tabular_data_experiments.results.rank_difference_utils import (
    graph_ranks,
    wilcoxon_holm,
)
from tabular_data_experiments.results.result import ConfigSplitResults, Results
from tabular_data_experiments.results.utils import (
    get_dataset_with_no_nans,
    get_order_of_labels_correlation,
    get_order_of_labels_median_collection_size,
    get_results_after_cutoff,
    get_total_time_per_method,
    make_incumbent_trajectory,
    random_shuffle,
    store_table_for_metric,
)


def rank_difference_plot(
    results: Results,
    method: str | list[str] | None = None,
    metric: str = "test_roc_auc",
    alpha: float = 0.05,
    tables_dir: Path | None = None,
) -> plt.Axes:
    """Plot the rank difference between methods. Significance is calculated using the Wilcoxon signed rank test and
    corrected using Holm's method.
    """
    labels, p_values, average_ranks, _ = get_average_ranks(results, method, metric, alpha, tables_dir)

    # plot on axes
    fig, ax = graph_ranks(
        avranks=[round(float(value), 2) for value in list(average_ranks.values)],
        names=average_ranks.keys(),
        p_values=p_values,
        cd=None,
        reverse=True,
        textspace=1.5,
        width=10,
    )

    return fig, ax


def get_average_ranks(results, method, metric, alpha, tables_dir) -> tuple[list[str], pd.DataFrame, pd.DataFrame, int]:
    r = results.at(
        method=method,
        metric=[metric],
    )
    dataset_key = "dataset" if "dataset" in r.df.columns else "task_id"

    df = r.df.groupby(["method", "fold"]).mean().groupby("method").mean().T

    # drop metric from multiindex
    df = df.droplevel(0)

    columns = df.columns
    # move dataset which is an index to a column
    df = df.reset_index()

    # concate mini dfs per method
    mini_dfs = []
    for column in columns:
        t_1_df = df[[dataset_key, column]].copy()
        t_1_df["method"] = column
        t_1_df.columns = ["Dataset", f"{metric}", "method"]
        mini_dfs.append(t_1_df)
    df = pd.concat(mini_dfs)
    methods = set(sorted(r.methods))

    labels = [FINAL_LABEL_NAMES[method] for method in methods]
    # replace method names
    # df = df.replace(methods, labels)
    df["method"] = df["method"].replace(methods, labels)
    # df.to_csv("Comparison.csv")
    # calculate p values
    p_values, average_ranks, num_datasets = wilcoxon_holm(
        df_perf=df, performance_metric_column_name=metric, alpha=alpha, tables_dir=tables_dir
    )
    return labels, p_values, average_ranks, num_datasets


def get_rank_per_collection_for_metrics(
    current_results: ConfigSplitResults,
    metrics: list[str],
    current_collections: dict[str, list[str]],
    tables_dir: Path,
    alpha: float = 0.05,
):
    results: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        # get incumbent results for metric
        current_incumbent_results = current_results.get_incumbent_results(
            metric=f"valid_{metric}", dataset=current_collections["all_datasets"]
        )
        rank_results = {}
        for collection in current_collections:
            collection_result_dir = tables_dir / collection
            collection_result_dir.mkdir(exist_ok=True, parents=True)
            try:
                labels, p_values, average_ranks, num_datasets = get_average_ranks(
                    current_incumbent_results.at(dataset=current_collections[collection]),
                    metric=f"test_{metric}",
                    method=None,
                    alpha=alpha,
                    tables_dir=collection_result_dir,
                )
                rank_results[collection] = average_ranks.astype(float)
                if num_datasets != len(current_collections[collection]):
                    print(
                        f"Only {num_datasets} datasets for {collection}, missing "
                        + f"{len(current_collections[collection]) - num_datasets} "
                        + "datasets"
                    )
                    continue
            except Exception as e:
                print(e)
                continue

        (tables_dir / "all").mkdir(exist_ok=True, parents=True)
        rank_results["all_datasets"] = get_average_ranks(
            current_incumbent_results.at(dataset=current_collections["all_datasets"]),
            metric=f"test_{metric}",
            method=None,
            alpha=alpha,
            tables_dir=tables_dir / "all",
        )[2].astype(float)
        rank_results = pd.DataFrame(rank_results)

        results[metric] = rank_results
    return results


def get_autorank_results(
    results: Results,
    method: str | list[str] | None = None,
    metric: str = "test_roc_auc",
    alpha: float = 0.05,
) -> Any:

    r = results.at(
        method=method,
        metric=[metric],
    )

    df = r.df.groupby(["method", "fold"]).mean().groupby("method").mean().T

    # drop metric from multiindex
    df = df.droplevel(0)

    # modify column names
    df.columns = [column if column not in FINAL_LABEL_NAMES else FINAL_LABEL_NAMES[column] for column in df.columns]

    result = autorank(df, alpha=alpha)
    return result


def cd_autorank_plot(
    results: Results,
    ax: plt.Axes,
    method: str | list[str] | None = None,
    metric: str = "test_roc_auc",
    alpha: float = 0.05,
    allow_insignificant: bool = True,
) -> plt.Axes:

    result = get_autorank_results(
        results=results,
        method=method,
        metric=metric,
        alpha=alpha,
    )

    ax = plot_stats(result, ax=ax, allow_insignificant=allow_insignificant)

    return ax


def incumbent_plot_with_method(
    results: ConfigSplitResults,
    ax: plt.Axes,
    dataset: str,
    method: str,
    opt_metric: str = "valid_roc_auc",
    plot_metric: str = "test_roc_auc",
    color: str = "blue",
    alpha: float = 0.05,
) -> plt.Axes:
    """plot incumbent plots for datasets"""
    r = results.at(
        method=method,
        dataset=dataset,
        metric=[opt_metric, plot_metric],
    )

    df = r.df

    df = make_incumbent_trajectory(df, opt_metric, plot_metric)
    df = df.reset_index()

    mean, min, max = random_shuffle(df)
    x = list(range(len(mean)))
    ax.plot(x, mean, label=FINAL_LABEL_NAMES[method], color=color, linewidth=1.5)
    ax.fill_between(
        x,
        min,
        max,
        color=color,
        alpha=alpha,
    )
    return ax


def incumbent_plot_on_dataset(
    results: ConfigSplitResults,
    ax: plt.Axes,
    dataset: str,
    methods: list[str] | None = None,
    opt_metric: str = "valid_roc_auc",
    plot_metric: str = "test_roc_auc",
    alpha: float = 0.05,
) -> plt.Axes:
    """plot incumbent plots for datasets"""
    color_palette = cc.glasbey_dark
    methods = methods if methods is not None else results.methods
    for method, color in zip(methods, color_palette):
        ax = incumbent_plot_with_method(
            results=results,
            ax=ax,
            dataset=dataset,
            method=method,
            opt_metric=opt_metric,
            plot_metric=plot_metric,
            color=color,
            alpha=alpha,
        )
    return ax


def plot_walltime_for_method(
    results: ConfigSplitResults | Results,
    method: str,
    ax: plt.Axes,
    fig: plt.Figure,
    bins: int = 100,
    color: str = "blue",
    alpha: float = 0.5,
    only_plot_axes: bool = False,
    log_scale: bool = True,
    fig_path: Path | None = None,
) -> plt.Axes:
    """plot the walltime distribution for a given method"""
    if fig_path is None:
        fig_path = Path("latest_results/plotting/walltime_hist")
    fig_path.mkdir(exist_ok=True, parents=True)
    walltimes = results.at(split=0, fold=0, method=method).df["total_walltime"]
    # insert 24 hrs in seconds to all nans
    walltimes = walltimes.fillna(24 * 60 * 60)
    times = walltimes.to_numpy().flatten()
    ax.hist(times, bins=bins, label=method, color=color, alpha=alpha)
    if not only_plot_axes:
        ax.set_xlabel("Total Walltime (s)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Total Walltime Distribution for {method}")
        if log_scale:
            ax.set_yscale("log")
        fig.show()
        fig.savefig(fig_path / f"{method}.png")
    return ax


def plot_walltime_for_method_per_dataset(
    results: ConfigSplitResults | Results,
    method: str,
    fig_size: tuple[float, float] = (10, 10),
    bins: int = 100,
    log_scale: bool = True,
    fig_path: Path | None = None,
) -> plt.Axes:
    """plot the walltime distribution for a given method"""
    if fig_path is None:
        fig_path = Path("latest_results/plotting/walltime_hist")
    fig_path.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=fig_size)
    for color, task_id in zip(cc.glasbey_dark, results.task_ids):
        walltimes = results.at(split=0, fold=0, dataset=task_id).df["total_walltime"]
        # insert 24 hrs in seconds to all nans
        walltimes = walltimes.fillna(24 * 60 * 60)
        times = walltimes.to_numpy().flatten()
        ax.hist(times, bins=bins, label=task_id, color=color, alpha=0.5)
    ax.set_xlabel("Total Walltime (s)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Total Walltime Distribution for {method}")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.1, 1.1), ncol=2)
    fig.show()
    fig.savefig(fig_path / f"{method}.png")
    return ax


def plot_rank_difference(
    datasets: list[str],
    results_dir: Path,
    metric: str,
    incumbent_results: Results,
    collection: str,
    caption: str | None = None,
) -> None:
    """
    Plot the rank difference diagram for a given metric and collection of datasets.

    Args:
        datasets(list[str]): datasets to plot
        results_dir(Path): results directory
        metric(str): metric to plot
        incumbent_results(Results): incumbent results
        collection(str): collection to plot
        caption(str): caption for the figure
    """

    figs_dir = results_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = rank_difference_plot(
        results=incumbent_results.at(dataset=datasets), metric=f"test_{metric}"
    )  # , tables_dir=tables_dir)
    # cd_autorank_plot(results=incumbent_results.at(dataset=dataset_collections[collection]), metric=f"test_{metric}", ax=ax, dropna=True) #, tables_dir=tables_dir)
    ax.set_title(f"Comparison test_{metric} Rank Difference Diagram on {collection} datasets", fontsize=26)
    if caption is not None:
        ax.text(
            0.5,
            -0.1,
            caption,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=26,
        )

    plt.show()
    fig.savefig(figs_dir / f"cd_diagram_test_{metric}_{collection}_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figs_dir / f"cd_diagram_test_{metric}_{collection}_comparison.png", dpi=300, bbox_inches="tight")


def plot_parallel_coordinates_plot(
    rank_results: pd.DataFrame,
    collection_mapping: dict[str, str],
    results_dir: Path,
    metric: str,
    colors=cc.glasbey_dark,
    title_suffix: str | None = None,
    figsize: tuple[float, float] = (24, 8),
    fontsize: int = 26,
    linewidth: int = 3,
    alpha: float = 0.8,
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] = (1.1, 1),
    cpu_models: list[str] | None = None,
    gpu_models: list[str] | None = None,
    models: list[str] | None = None,
):
    """
    Plot parallel coordinates plot for the rank results, includes CPU and GPU models
    that are also part of the final models

    Args:
        rank_results(pd.DataFrame): rank results
        collection_mapping(dict[str, str]): collection mapping
        results_dir(Path): results directory
        colors(list[str]): colors to use
        title_suffix(str): title suffix

    Returns:
        None
    """
    if cpu_models is None:
        cpu_models = CPU_MODELS
    if gpu_models is None:
        gpu_models = GPU_MODELS
    if models is None:
        models = FINAL_MODELS
    transformed_cpu_models = [FINAL_LABEL_NAMES[model] for model in set(cpu_models).intersection(models)]
    transformed_gpu_models = [FINAL_LABEL_NAMES[model] for model in set(gpu_models).intersection(models)]
    colors = [colors[i] for i, _ in enumerate(transformed_cpu_models + transformed_gpu_models)]
    colors_cpu = colors[: len(transformed_cpu_models)]
    colors_gpu = colors[len(transformed_cpu_models) :]
    title_suffix = "" if title_suffix is None else title_suffix
    results_dir.mkdir(exist_ok=True, parents=True)
    rank_results = rank_results.reset_index(names=["method", "collection"])
    rank_results.columns = [collection_mapping.get(c, c) for c in rank_results.columns]
    # Visualise the rank of the 4 models using parallel coordinates plots
    fig, ax_plot = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    rank_results_cpu = rank_results[rank_results["method"].isin(transformed_cpu_models)]
    rank_results_gpu = rank_results[rank_results["method"].isin(transformed_gpu_models)]
    pd.plotting.parallel_coordinates(
        rank_results_cpu,
        class_column="method",
        color=colors_cpu,
        ax=ax_plot,
        axvlines=False,
        linewidth=linewidth,
        alpha=alpha,
        linestyle="--",
    )
    pd.plotting.parallel_coordinates(
        rank_results_gpu,
        class_column="method",
        color=colors_gpu,
        ax=ax_plot,
        axvlines=False,
        linewidth=linewidth,
        alpha=alpha,
        linestyle="-",
    )
    ax_plot.set_xticklabels(ax_plot.get_xticklabels(), rotation=90, fontsize=fontsize)
    ax_plot.legend(*ax_plot.get_legend_handles_labels(), bbox_to_anchor=bbox_to_anchor, ncol=ncol, fontsize=fontsize)
    # ax_legend.axis("off")
    # ax.set_xlabel("Collection", fontsize=20)
    ax_plot.set_ylabel("Average Rank", fontsize=fontsize)
    ax_plot.set_title(
        f"Average Rank of Models on Datasets from Different Collections ({title_suffix})", fontsize=fontsize
    )
    ax_plot.grid(True, alpha=alpha - 0.2)
    fig.show()
    fig.savefig(results_dir / f"parallel_coordinates_{metric}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / f"parallel_coordinates_{metric}.png", dpi=300, bbox_inches="tight")


def plot_cd_diagram_on_dataset_collections(
    metrics: list[str],
    dataset_collections: dict[str, list[str]],
    results_dir: Path,
    current_results: ConfigSplitResults,
    caption: str | None = None,
    only_ranks: bool = False,
    try_plot: bool = True,
) -> None | Any:
    """
    Plot the cd diagram for a given metric and collection of datasets.

    Args:
        metrics(list[str]): metrics to plot
        dataset_collections(dict[str, list[str]]): dictionary of dataset collections
        results_dir(Path): results directory
        current_results(ConfigSplitResults): current results
        caption(str): caption for the figure
        only_ranks(bool): only get ranks of the comparison
        try_plot(bool): try to plot the figure

    Returns:
        None
    """
    for metric in metrics:
        incumbent_results = current_results.get_incumbent_results(
            metric=f"valid_{metric}",
        )
        for collection in dataset_collections:
            print(f"Plotting {metric} for {collection}")
            # if only getting ranks of the comparison
            if only_ranks:
                try:
                    result = get_autorank_results(
                        incumbent_results.at(dataset=dataset_collections[collection]),
                        metric=f"test_{metric}",
                    )
                except Exception as e:
                    if try_plot:
                        print(e)
                        continue
                    else:
                        raise e
                return result
            try:
                store_table_for_metric(metric, collection, incumbent_results, dataset_collections, results_dir)
            except Exception as e:
                if try_plot:
                    print(e)
                    continue
                else:
                    raise e

            try:
                plot_rank_difference(
                    datasets=dataset_collections[collection],
                    results_dir=results_dir,
                    metric=metric,
                    incumbent_results=incumbent_results,
                    caption=caption,
                    collection=collection,
                )
            except Exception as e:
                if try_plot:
                    print(e)
                    continue
                else:
                    raise e


# Plot incument trajectory on 361481 for all methods o
def plot_incumbent_on_dataset_id(
    results: ConfigSplitResults,
    split: int = 0,
    fold: int = 0,
    dataset_id: str = "361481",
    metric: str = "roc_auc",
    fig_path: Path | None = None,
    bbox_to_anchor: tuple[float, float] = (1.1, 1.1),
    log_scale: bool = True,
) -> None:
    """
    Plot incumbent trajectory on a dataset.

    Args:
        results(ConfigSplitResults): Results to plot.
        split(int): Split number. Defaults to 0.
        fold(int): Fold number. Defaults to 0.
        dataset_id(str): Dataset id. Defaults to "361481".
        fig_path(Path | None): Path to save figure. Defaults to None.
        bbox_to_anchor(tuple[float, float]): Bbox to anchor legend. Defaults to (1.1, 1.1).
        log_scale(bool): Whether to use log scale for x axis. Defaults to True.
    """
    fig, ax = plt.subplots()
    incumbent_plot_on_dataset(
        results=results.at(split=split, fold=fold),
        dataset=dataset_id,
        ax=ax,
        opt_metric=f"valid_{metric}",
        plot_metric=f"test_{metric}",
    )
    ax.set_title(f"Incumbent Trajectory on Dataset {dataset_id}")
    ax.set_xlabel("Wallclock Time (s)")
    ax.set_ylabel(f"Test {metric.capitalize()}")
    ax.legend(bbox_to_anchor=bbox_to_anchor)
    if log_scale:
        ax.set_xscale("log")
    fig.show()
    if fig_path is not None:
        fig_path.mkdir(exist_ok=True, parents=True)
        fig.savefig(fig_path / f"incumbent_trajectory_{dataset_id}.pdf", bbox_inches="tight")
        fig.savefig(fig_path / f"incumbent_trajectory_{dataset_id}.png", bbox_inches="tight")


def plot_pcp_metrics(
    rank_results: dict[str, pd.DataFrame],
    metrics: list[str],
    collection_mapping: dict[str, str],
    results_dir: Path,
) -> None:
    """
    Plot parallel coordinates plot for the rank results, includes CPU and GPU models
    that are also part of the final models

    Args:
        rank_results(dict[str, pd.DataFrame]): rank results
        metrics(list[str]): metrics to plot
        collection_mapping(dict[str, str]): collection mapping
        results_dir(Path): results directory

    Returns:
        None
    """

    for metric in metrics:
        current_rank_results = rank_results[metric].copy()
        current_rank_results = current_rank_results.dropna(axis=1)
        if current_rank_results.shape != rank_results[metric].shape:
            print(f"Warning: {metric} has nan values")
            # print missing collections
            missing_collections = set(rank_results[metric].columns).difference(set(current_rank_results.columns))
            print(f"Missing collections: {missing_collections}")
        tau_order = get_order_of_labels_correlation(current_rank_results)
        median_order = get_order_of_labels_median_collection_size(current_rank_results)
        plot_parallel_coordinates_plot(
            current_rank_results[tau_order],
            collection_mapping,
            results_dir / "mse_tau_order",
            title_suffix=f"Ordered by decreasing Kendall's Tau and MSE {metric=}",
            metric=metric,
        )
        plot_parallel_coordinates_plot(
            current_rank_results[median_order],
            collection_mapping,
            results_dir / "median_order",
            title_suffix=f"Ordered by increasing Median Dataset Size {metric=}",
            metric=metric,
        )


def plot_rank_random_sets(
    tasks: list[str],
    all_res: pd.DataFrame,
    frequency_rank: pd.DataFrame,
    results_dir: Path,
    metric: str,
    n_rep: int = 100,
    y_lim: Optional[tuple[float, float]] = None,
    text_y_offset: Optional[dict[str, dict[str, int]]] = None,
    random_shuffle_alpha: float = 0.06,
):
    # Plotting
    fig, ax = plt.subplots(figsize=(16, 9))
    # from https://icolorpalette.com/color?q=red
    x_steps = np.arange(0, len(tasks))

    for model in all_res.index:
        color = MODELS_TO_COLORS[model]
        # Thin line for each rank of randomly shuffle subsets
        for r in range(n_rep):
            ax.plot(
                x_steps, all_res.xs(f"rep_{r}", level="rep", axis=1).loc[model], c=color, linewidth=0.05, alpha=random_shuffle_alpha
            )

        # Plot average rank and rank by frequency bold
        average = [all_res.loc[model].xs(f"size_{s}", level="size").mean(axis=0) for s in x_steps]
        ax.plot(
            x_steps,
            average,
            c=color,
            zorder=99,
            linestyle=":",
            linewidth=2.5,
        )
        ax.plot(
            x_steps,
            frequency_rank.loc[model],
            c=color,
            zorder=99,
            linewidth=2.5,
        )

        # Plot model name to the right w/ some offset
        offset = 0
        if text_y_offset is not None:
            offset = text_y_offset.get(metric, {}).get(model, 0)
        ax.text(len(tasks) + 5, average[-1] + offset, FINAL_LABEL_NAMES[model], fontsize=26, c=color)

    # Fake legend
    ax.plot([0], [0], c="k", linewidth=0.05, label="Random")
    ax.plot([0], [0], c="k", linewidth=2.5, label="Sorted by frequency")
    ax.plot([0], [0], c="k", linewidth=2.5, linestyle=":", label="Average")
    ax.legend(fontsize=20)

    # Add some description
    #ax.set_title(metric, fontsize=26)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim([frequency_rank.min().min(), frequency_rank.max().max()])
    ax.set_xlim([5, x_steps[-1]])
    ax.set_ylabel(f"Average Rank ({metric})", fontsize=26)
    ax.set_xlabel("#datasets", fontsize=26)
    # plt.xscale("log")
    ax.set_xticklabels([int(tl) for tl in ax.get_xticks()], fontsize=26)
    ax.set_yticklabels([int(tl) for tl in ax.get_yticks()], fontsize=26)
    # Voila
    fig.tight_layout()
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(results_dir / f"rank_random_sets_{metric}.pdf")
    fig.show()


def plot_hardness_vs_age(
    hardness: pd.DataFrame,
    criteria: str,
    output_dir: Path,
):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        data=hardness,
        x="age",
        y="hardness",
        # hue="0",
        palette="viridis",
        ax=ax,
        s=100,
        alpha=0.8,
        legend=False,
    )

    # kendall tau
    result = kendalltau(hardness["age"], hardness["hardness"], alternative="greater", variant="c")
    # One could also get the confidence intervals if required
    # from scipy.stats import bootstrap
    # ref = bootstrap(
    #     (hardness["age"].to_numpy(), hardness["hardness"].to_numpy()),
    #     lambda x, y: kendalltau(x, y, alternative="greater").statistic,
    #     paired=True,
    # )

    # add correlation and pvalue to plot
    ax.text(0.05, 0.95, f"Kendall's tau: {result.correlation:.2f}", transform=ax.transAxes, fontsize=26)
    ax.text(0.05, 0.90, f"p-value: {result.pvalue:.2f}", transform=ax.transAxes, fontsize=26)
    ax.set_xlabel("Dataset Creation Year", fontsize=26)
    ax.set_ylabel(f"Hardness ({criteria.capitalize()})", fontsize=26)
    # ax.set_title("Hardness vs Dataset Creation Year", fontsize=20)
    fig.tight_layout()

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    print("Plotting to", output_dir)
    fig.savefig(output_dir / f"hardness_vs_age-{criteria}.pdf")
    fig.savefig(output_dir / f"hardness_vs_age-{criteria}.png")


#  visualize
def plot_n_configs_per_dataset(
    n_configs_per_dataset: pd.DataFrame,
    output_dir: Path,
):
    for method in n_configs_per_dataset.index:
        temp_df = n_configs_per_dataset.loc[method][n_configs_per_dataset.loc[method] > 0].copy()
        # if n_datasets > 20 increase figsize
        if len(temp_df) > 20:
            fig, ax = plt.subplots(figsize=(30, 10))
        else:
            fig, ax = plt.subplots(figsize=(10, 5))
        # drop all datasets for the model with 0 failed configs
        temp_df.plot.bar(ax=ax, label=method, alpha=0.5)
        ax.set_ylabel("Number of failed configurations")
        ax.set_xlabel("Method")
        ax.legend()
        ax.set_title(
            f"Number of failed configurations per dataset for {method} total {len(temp_df)} datasets with failed configs"
        )
        ax.set_yticklabels(ax.get_yticks(), rotation=90)
        fig.show()
        fig.savefig(output_dir / "failed_configs_per_dataset_{method}.pdf", bbox_inches="tight")


def plot_total_walltime_per_method_per_cutoff_per_fold(
    current_results: ConfigSplitResults,
    cutoffs: list,
    plot_dir: Path,
    folds: list[int],
    n_configs: int = 100,
    n_splits: int = 5,
):
    plot_dir.mkdir(exist_ok=True, parents=True)
    results_folds = {}
    for fold in folds:
        results_folds[fold] = {}
        fold_dir = plot_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        for cutoff in cutoffs:
            fig, ax = plt.subplots(figsize=(10, 10))
            df = get_total_time_per_method(
                get_results_after_cutoff(current_results, cutoff=cutoff),
                n_configs=n_configs,
                n_folds=fold,
                n_splits=n_splits,
            )
            results_folds[fold][cutoff] = df
            ax = df.plot.bar(ax=ax)
            ax.set_ylabel("Total Walltime (days)")
            ax.set_xlabel("Method")
            ax.set_title(
                f"Total Walltime per method for cutoff {cutoff} for {n_configs} configs, {fold} folds and {n_splits} splits"
            )
            # set yticks as int
            ax.set_yticklabels([f"{int(tick):,}" for tick in ax.get_yticks().tolist()])
            # set xticks rotation
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            fig.show()
            fig.savefig(fold_dir / f"total_walltime_per_method_cutoff_{cutoff}.png", dpi=300, bbox_inches="tight")
            fig.savefig(fold_dir / f"total_walltime_per_method_cutoff_{cutoff}.pdf", dpi=300, bbox_inches="tight")

    return results_folds


# visualize


def plot_embeddings_collections(embed, y, title="t-SNE of dataset collections"):
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(
        x=embed[:, 0],
        y=embed[:, 1],
        hue=y if y is not None else None,
        ax=ax,
        palette="tab20",
        legend="full",
        s=100,
        alpha=0.8,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title) if title is not None else None
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    return fig, ax


def plot_embeddings_one_figure(
    collection_mapping: dict[str, str],
    dataset_ids: dict[str, int],
    embed: pd.DataFrame,
    title: str = "t-SNE of datasets",
    alpha: float = 0.5,
    output_file: Path = None,
):
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    colors = cc.glasbey_dark

    fig, ax = plt.subplots(figsize=(7, 9))

    for color, collection in zip(colors, collection_mapping):

        dataset_ids_collection = dataset_ids[collection]
        # add temp bool flag to indicate if dataset is in collection
        temp_df = embed.copy()
        in_collection = embed["dataset_id"].isin(dataset_ids_collection)
        sns.scatterplot(
            x="t-SNE 1",
            y="t-SNE 2",
            data=temp_df[in_collection],
            ax=ax,
            color=color,
            legend="full",
            s=100,
            alpha=alpha,
            label=collection_mapping[collection],
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title) if title is not None else None
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    fig.tight_layout()
    fig.show()
    if output_file is not None:
        fig.savefig(f"{output_file}.png", bbox_inches="tight")
        fig.savefig(f"{output_file}.png", bbox_inches="tight")


def plot_embeddings_datasets(
    collection_mapping: dict[str, str],
    dataset_ids: dict[str, int],
    embed: pd.DataFrame,
    title: str = "t-SNE of datasets",
    alpha: float = 0.5,
    output_file: Path = None,
):
    sns.set_theme()
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

    for i, collection in enumerate(collection_mapping):

        dataset_ids_collection = dataset_ids[collection]
        # add temp bool flag to indicate if dataset is in collection
        temp_df = embed.copy()
        temp_df["in_collection"] = embed["dataset_id"].isin(dataset_ids_collection)
        # Map True False to in collection and not in collection
        temp_df["in_collection_label"] = temp_df["in_collection"].map(
            {True: "in_collection", False: "not_in_collection"}
        )
        fig, ax = plt.subplots(figsize=(7, 9))
        color = cc.glasbey_dark
        for label, color in zip([True, False], color):
            sns.scatterplot(
                x="t-SNE 1",
                y="t-SNE 2",
                hue="in_collection_label",
                data=temp_df[temp_df["in_collection"] == label],
                ax=ax,
                palette=[color],
                legend="full",
                s=100,
                alpha=alpha,
            )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(title + f"_{collection}") if title is not None else None
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        fig.tight_layout()
        fig.show()
        if output_file is not None:
            fig.savefig(f"{output_file}_{collection}.png", bbox_inches="tight")
            fig.savefig(f"{output_file}_{collection}.png", bbox_inches="tight")
