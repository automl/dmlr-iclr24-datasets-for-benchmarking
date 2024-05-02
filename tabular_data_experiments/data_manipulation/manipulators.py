"""Defines functions that manipulate the data in some way."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing_extensions import TypeAlias

DataManipulaterType: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame, NDArray[np.bool_], int],
    Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], Dict[str, Any]],
]

PREPROCESS_SEED = 0


def remove_high_cardinality(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], threshold: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """Remove columns with high cardinality"""
    high_cardinality_mask = np.array(X.nunique() > threshold) * categorical_indicator
    print(high_cardinality_mask)
    high_cardinality_cols = X.columns[high_cardinality_mask]
    print(f"high cardinality columns: {high_cardinality_cols}")
    n_high_cardinality = sum(high_cardinality_mask)
    X = X.drop(high_cardinality_cols, axis=1)
    print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    # update categorical mask
    categorical_indicator = [
        categorical_indicator[i] for i in range(len(categorical_indicator)) if not high_cardinality_mask[i]
    ]
    return X, y, np.array(categorical_indicator), {"n_high_cardinality": n_high_cardinality}


def remove_pseudo_categorical(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], threshold: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """Remove columns where most values are the same"""
    pseudo_categorical_cols_mask = X.nunique() < threshold * (~categorical_indicator)
    print("Removed {} pseudo-categorical features".format(sum(pseudo_categorical_cols_mask)))
    X = X.drop(X.columns[pseudo_categorical_cols_mask], axis=1)
    return X, y, categorical_indicator, {"num_pseudo_categorical_cols": sum(pseudo_categorical_cols_mask)}


def remove_rows_with_missing_values(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    missing_rows_mask = pd.isnull(X).any(axis=1)
    print("Removed {} rows with missing values on {} rows".format(sum(missing_rows_mask), X.shape[0]))
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return X, y, categorical_indicator, {"num_missing_rows": sum(missing_rows_mask)}


def remove_missing_values(
    X: pd.DataFrame,
    y: pd.DataFrame,
    categorical_indicator: NDArray[np.bool_],
    threshold=0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    print("Removed {} columns with missing values on {} columns".format(sum(missing_cols_mask), X.shape[1]))
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    # update categorical mask
    categorical_indicator = np.array(
        [categorical_indicator[i] for i in range(len(categorical_indicator)) if not missing_cols_mask[i]]
    )
    X, y, categorical_indicator, additional_info = remove_rows_with_missing_values(X, y, categorical_indicator)
    return X, y, categorical_indicator, {"num_missing_cols": sum(missing_cols_mask), **additional_info}


def balance_binarize(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """
    Balance the dataset by undersampling the majority class. In case of multiclass
    classification, the two most numerous classes are balanced."""
    rng = np.random.default_rng(PREPROCESS_SEED)
    if len(np.unique(y)) == 1:
        raise ValueError("Cannot balance dataset with only one class")
    # collect indices of each class
    indices = [(y == i) for i in np.unique(y)]
    # sort classes by number of samples ascending
    sorted_classes = np.argsort(
        list(map(sum, indices))
    )  # in case there are more than 2 classes, we take the two most numerous
    n_samples_min_class = sum(indices[sorted_classes[-2]])
    # sample the majority class
    indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
    indices_min_class = np.where(indices[sorted_classes[-2]])[0]
    total_indices = np.concatenate((indices_max_class, indices_min_class))
    y = y[total_indices]
    indices_first_class = y == sorted_classes[-1]
    indices_second_class = y == sorted_classes[-2]
    y[indices_first_class] = 0
    y[indices_second_class] = 1

    return X.iloc[total_indices], y, categorical_indicator, {}


def subsample_rows(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], threshold: int = 10000, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """Subsample the dataset if it is too large"""
    if X.shape[0] > threshold:
        rng = np.random.default_rng(PREPROCESS_SEED)
        indices = rng.choice(X.shape[0], threshold, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    return X, y, categorical_indicator, {"num_samples": X.shape[0]}


def subsample_columns(
    X: pd.DataFrame, y: pd.DataFrame, categorical_indicator: NDArray[np.bool_], threshold: int = 10, **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
    """Subsample the dataset if it is too large"""
    if X.shape[1] > threshold:
        rng = np.random.default_rng(PREPROCESS_SEED)
        indices = rng.choice(X.shape[1], threshold, replace=False)
        X = X.iloc[:, indices]
        categorical_indicator = categorical_indicator[indices]
    return X, y, categorical_indicator, {"num_features": X.shape[1]}


# def change_imbalance(
#         X: pd.DataFrame,
#         y: pd.DataFrame,
#         categorical_indicator: NDArray[np.bool_],
#         threshold: int = 0.5,
#         **kwargs
#     ) -> Tuple[pd.DataFrame, pd.DataFrame, NDArray[np.bool_], dict[str, Any]]:
#     """Change the imbalance of the dataset. Here threshld is the number of samples in the majority class"""
#     rng = np.random.RandomState(0)
#     if len(np.unique(y)) == 1:
#         # return empty arrays
#         return np.array([]), np.array([])
#     # collect indices of each class
#     indices = [(y == i) for i in np.unique(y)]
#     # sort classes by number of samples ascending
#     sorted_classes = np.argsort(
#         list(map(sum, indices)))
#     n_samples_max_class = sum(indices[sorted_classes[-1]])
#     n_samples_other_classes = sum([sum(indices[i]) for i in sorted_classes[:-1]])

#     if n_samples_max_class/X.shape[0] > threshold:
#         # subsample the majority class
#         required_max_class_samples = int(threshold*n_samples_other_classes) -
#     else:
#         # subsample other classes
#     # sample the majority class
#     indices_max_class = rng.choice(np.where(indices[sorted_classes[-1]])[0], n_samples_min_class, replace=False)
#     indices_min_class = np.where(indices[sorted_classes[-2]])[0]
#     total_indices = np.concatenate((indices_max_class, indices_min_class))
#     y = y[total_indices]

#     return X.iloc[total_indices], y, categorical_indicator, {}
