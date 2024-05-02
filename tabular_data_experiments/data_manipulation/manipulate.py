"""Manipulate the data."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from tabular_data_experiments.data_manipulation import DataManipulators
from tabular_data_experiments.utils.types import TargetDatasetType


def manipulate_data(
    X: TargetDatasetType,
    y: TargetDatasetType,
    categorical_indicator: NDArray[np.bool_],
    manipulators: list[str],
    **manipulator_kwargs: dict[str, Any],
) -> Tuple[TargetDatasetType, TargetDatasetType, NDArray[np.bool_], dict[str, Any]]:
    """Manipulate the data.

    Args:
        data (DatasetType): _description_
        manipulators (list[str]): _description_

    Returns:
        DatasetType: _description_
    """
    manipulators_additional_info = {}
    for manipulator in manipulators:
        kwargs = {
            k.replace(f"{manipulator}__", ""): v
            for k, v in manipulator_kwargs.items()
            if k.startswith(f"{manipulator}__")
        }
        (X, y, categorical_indicator, manipulator_additional_info) = DataManipulators[manipulator](
            X, y, categorical_indicator, **kwargs
        )
        manipulators_additional_info.update(manipulator_additional_info)
    return X, y, categorical_indicator, manipulators_additional_info
