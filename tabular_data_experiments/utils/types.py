"""Module containing custom types for tabular data experiments.""" ""
from typing import Callable, TypeVar

from pathlib import Path

import pandas as pd
from typing_extensions import TypeAlias

InputDatasetType = TypeVar("InputDatasetType", bound=pd.DataFrame)
TargetDatasetType = TypeVar("TargetDatasetType", pd.DataFrame, pd.Series)
PathType = TypeVar("PathType", str, Path)
