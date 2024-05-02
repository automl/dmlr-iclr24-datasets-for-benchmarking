"""Base class for data loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


class BaseLoader(ABC):
    """
    Base class for data loaders.
    """

    def __init__(
        self,
    ) -> None:
        """Base class for data loaders."""
        pass

    @classmethod
    def create(self, loader_type: str, loader_kwargs: dict[str, Any]) -> "BaseLoader":
        """Create a data loader."""
        if loader_type == "openml":
            from tabular_data_experiments.data_loaders.openml import OpenMLLoader

            return OpenMLLoader(**loader_kwargs)
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")

    @abstractmethod
    def load_data(self, **kwargs) -> Tuple[InputDatasetType, TargetDatasetType, NDArray[np.bool_], list[str]]:
        """Load data from the given path."""
        pass

    def get_data(self, **kwargs) -> Tuple[InputDatasetType, TargetDatasetType, NDArray[np.bool_], list[str]]:
        """Get the data."""
        return self.load_data(**kwargs)
