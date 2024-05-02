""" Data utilities for tabular data experiments. """
from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.utils.multiclass import type_of_target


def get_required_dataset_info(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    categorical_indicator: Iterable[bool] | None = None,
    old_categorical_columns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Get the dataset properties. Currently it is specific to
    reg cocktails model.

    Args:
        X (DatasetType): _description_
        y (DatasetType): _description_
        categorical_indicator (NDArray[np.bool_]): _description_

    Returns:
        dict: _description_
    """
    properties = {}
    properties["output_type"] = type_of_target(y)
    properties["task_type"] = "tabular_classification"
    properties["categorical_columns"] = []
    properties["numerical_columns"] = []
    n_classes = len(np.unique(y))
    properties["n_classes"] = n_classes
    properties["output_shape"] = n_classes if n_classes > 2 else 1
    properties["feature_names"] = X.columns.tolist()
    if categorical_indicator is None:
        categorical_indicator = [False] * X.shape[1]
        X = X.infer_objects()
        for i, column in enumerate(X.columns):
            if X[column].dtype.name in ["category", "bool", "object"]:
                categorical_indicator[i] = True
            if old_categorical_columns is not None and column in old_categorical_columns:
                categorical_indicator[i] = True

    for column, is_categorical in zip(X.columns, categorical_indicator):
        if is_categorical:
            properties["categorical_columns"].append(column)
        else:
            properties["numerical_columns"].append(column)
    properties["categorical_indicator"] = categorical_indicator
    properties["n_categories_per_cat_col"] = [
        len(X[col].dropna().unique()) for col in properties["categorical_columns"]
    ]
    properties["issparse"] = False
    properties["all_nan_col_idx"] = [i for i, col in enumerate(X.columns) if X[col].isna().all()]
    return properties


# used for refitting.
def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    task: openml.OpenMLTask,
    fold_number: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test."""
    train_indices, test_indices = task.get_train_test_split_indices(fold=fold_number)

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test
