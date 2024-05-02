"""Class to load data from OpenML. Given a dataset id, it downloads the data and splits it into train and test. It is a base class of base_loader.py.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import openml
import pandas as pd
from numpy.typing import NDArray
from sklearn.calibration import LabelEncoder

from tabular_data_experiments.data_loaders.base_loader import BaseLoader
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class OpenMLLoader(BaseLoader):
    """
    Class to load data from OpenML. Given a dataset id, it downloads the data.
    It is a child class of base_loader.py.

    Args:
        BaseLoader (_type_): _description_
    """

    def __init__(self, *, dataset_id: int | None = None, task_id: int | None = None) -> None:
        super().__init__()
        self.label_encoder = LabelEncoder()
        if dataset_id is None and task_id is None:
            raise ValueError("Either dataset_id or task_id must be provided.")
        if task_id is not None:
            self.task = self._get_task(task_id=task_id)
            self.dataset_id = self.task.dataset_id
        elif dataset_id is not None:
            self.dataset_id = dataset_id

    def _get_task(self, task_id: int, **kwargs) -> openml.tasks.Task:
        """
        Get the task from the task id.

        Args:
            task_id: The task id.

        Returns:
            The task.
        """
        return openml.tasks.get_task(task_id=task_id, **kwargs)

    def load_data(self, **kwargs) -> Tuple[InputDatasetType, TargetDatasetType, NDArray[np.bool_], List[str]]:
        """Get the data from the dataset id.

        Returns:
           Tuple[DatasetType, DatasetType, List[bool], List[str]]: _description_
        """

        assert self.dataset_id is not None
        target_name = self.task.target_name  # type: ignore
        X, y, categorical_indicator, attribute_names = self._get_data_from_dataset_id(
            dataset_id=self.dataset_id, target_attribute=target_name, **kwargs
        )
        y = pd.Series(self.label_encoder.fit_transform(y))
        return X, y, categorical_indicator, attribute_names

    @classmethod
    def _get_data_from_dataset_id(
        cls,
        target_attribute: str,
        dataset_id: int | None,
        data_format: str = "dataframe",
    ) -> Tuple[InputDatasetType, TargetDatasetType, NDArray[np.bool_], List[str]]:
        """Loads the data from OpenML and splits it into train and test.

        Args:
            dataset_id (int): _description_
        """

        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format=data_format, target=target_attribute
        )
        return X, y, np.array(categorical_indicator), attribute_names
