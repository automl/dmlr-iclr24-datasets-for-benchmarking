from __future__ import annotations

import pandas as pd


def augment_categories(X: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the categories. Adds 1 to all columns, which ensures that nans or new
    categoris are set to 0 and all the seen categories start from 1. Since, ordinal
    encoder outputs -1 for unknown values and missing values. This is done regardless
    of the presence of missing values in the training set. Negative values can not be
    handled by pytorch nn.Embedding.

    Note:
    To test for dataset where this is needed, use task_id 361503:
    https://www.openml.org/t/361503, fold 0 and split 0
    """
    # add lowest negative value to all features with negative values
    X += 1
    return X
