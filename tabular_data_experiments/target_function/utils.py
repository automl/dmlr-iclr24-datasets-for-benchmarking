"""Utility functions for HPO."""
from __future__ import annotations

from typing import Any, NamedTuple

from enum import IntEnum

from typing_extensions import TypedDict


class StatusType(IntEnum):
    """Class to define status types of configs."""

    SUCCESS = 1
    CRASHED = 2


class TargetFunctionInfo(TypedDict):
    """Additional information about a target function call."""

    status: StatusType
    train_loss: float | None
    valid_loss: float | None
    test_loss: float | None
    average_split_fit_time: float | None
    average_split_predict_time: float | None
    average_split_predict_time_valid: float | None
    average_split_predict_time_test: float | None
    average_split_preprocess_time_train: float | None
    average_split_preprocess_time_valid: float | None
    average_split_preprocess_time_test: float | None
    total_walltime: float | None
    config: dict[str, Any]
    n_splits: int | None
    train_metrics: dict[str, float] | dict[str, dict[str, float]] | None
    valid_metrics: dict[str, float] | dict[str, dict[str, float]] | None
    test_metrics: dict[str, float] | dict[str, dict[str, float]] | None
    additional_info: dict[str, Any] | None


class TargetFunctionResult(NamedTuple):
    """Result of a target function call."""

    loss: float | None
    info: TargetFunctionInfo
