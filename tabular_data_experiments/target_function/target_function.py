"""Target function for HPO experiments."""
from __future__ import annotations

from typing import Any

import time
from pathlib import Path

import numpy as np
import torch
from ConfigSpace import Configuration

from tabular_data_experiments.metrics import (
    Scorer,
    calculate_loss,
    calculate_scores,
    get_scorer,
)
from tabular_data_experiments.models.base_model import BaseModel
from tabular_data_experiments.target_function.utils import (
    StatusType,
    TargetFunctionInfo,
    TargetFunctionResult,
)
from tabular_data_experiments.utils.types import InputDatasetType, TargetDatasetType


def train_single_split(
    X_train: InputDatasetType,
    y_train: TargetDatasetType,
    model_name: str,
    dataset_properties: dict[str, Any],
    config: Configuration,
    model_kwargs: dict[str, Any],
    optimize_metric: Scorer,
    additional_metrics: list[Scorer] | None = None,
    X_valid: InputDatasetType | None = None,
    y_valid: TargetDatasetType | None = None,
    X_test: InputDatasetType | None = None,
    y_test: TargetDatasetType | None = None,
    store_preds: bool = False,
    preds_dir: Path | None = None,
) -> TargetFunctionResult:
    """Train a model on a single split of the data."""
    start_time = time.time()

    cuda = torch.cuda.is_available()
    print(f"Running on cuda: {cuda}")
    if cuda:
        print(f"Available cuda devices: {torch.cuda.device_count()}")
        print(f"Current cuda device: {torch.cuda.current_device()}")
        print(f"Current cuda device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        torch.cuda.empty_cache()

    model = BaseModel.create(
        model_name, config=config, dataset_properties=dataset_properties, model_kwargs=model_kwargs
    )

    preprocess_start_time = time.time()
    X_train = model.preprocess_data(X_train)
    preprocess_duration = time.time() - preprocess_start_time
    fit_start_time = time.time()

    preprocess_duration_valid = None
    if X_valid is not None and y_valid is not None:
        preprocess_start_time = time.time()
        X_valid = model.preprocess_data(X_valid, training=False)
        preprocess_duration_valid = time.time() - preprocess_start_time

    model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

    fit_duration = time.time() - fit_start_time
    predict_start_time = time.time()
    train_preds = model.predict_proba(X_train)
    predict_duration = time.time() - predict_start_time
    if store_preds:
        assert preds_dir is not None
        preds_dir.mkdir(parents=True, exist_ok=True)
        np.save(preds_dir / "train_preds.npy", train_preds)

    train_loss = calculate_loss(y_train, train_preds, optimize_metric)

    if additional_metrics is not None:
        train_metrics = calculate_scores(y_train, train_preds, additional_metrics)

    valid_loss = None
    valid_metrics = None
    valid_start_time = None
    if X_valid is not None and y_valid is not None:
        valid_start_time = time.time()
        valid_preds = model.predict_proba(X_valid)
        valid_predict_duration = time.time() - valid_start_time
        if store_preds:
            np.save(preds_dir / "valid_preds.npy", valid_preds)
        valid_loss = calculate_loss(y_valid, valid_preds, optimize_metric)
        if additional_metrics is not None:
            valid_metrics = calculate_scores(y_valid, valid_preds, additional_metrics)

    test_loss = None
    test_metrics = None
    test_start_time = None
    preprocess_duration_test = None
    if X_test is not None and y_test is not None:
        preprocess_start_time = time.time()
        X_test = model.preprocess_data(X_test, training=False)
        preprocess_duration_test = time.time() - preprocess_start_time
        test_start_time = time.time()
        test_preds = model.predict_proba(X_test)
        test_predict_duration = time.time() - test_start_time
        if store_preds:
            np.save(preds_dir / "test_preds.npy", test_preds)

        test_loss = calculate_loss(y_test, test_preds, optimize_metric)
        if additional_metrics is not None:
            test_metrics = calculate_scores(y_test, test_preds, additional_metrics)

    duration = time.time() - start_time
    print(
        f"Finished fold with train loss: {train_loss}, "
        f"valid loss: {valid_loss}, test loss: {test_loss}, duration: {duration}"
    )

    cost = valid_loss if valid_loss is not None else train_loss
    info = TargetFunctionInfo(
        train_loss=train_loss,
        valid_loss=valid_loss,
        test_loss=test_loss,
        status=StatusType.SUCCESS,
        average_split_fit_time=fit_duration,
        average_split_predict_time=predict_duration,
        average_split_predict_time_test=test_predict_duration if test_start_time is not None else None,
        average_split_predict_time_valid=valid_predict_duration if valid_start_time is not None else None,
        average_split_preprocess_time_train=preprocess_duration,
        average_split_preprocess_time_valid=preprocess_duration_valid,
        average_split_preprocess_time_test=preprocess_duration_test,
        total_walltime=duration,
        config=model.config,
        n_splits=1,
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        additional_info={**model.get_additional_run_info(), **model.get_model_description()},
    )

    result = TargetFunctionResult(cost, info)

    return result


class TargetFunction(object):
    """Target function for HPO experiments."""

    def __init__(
        self,
        X_train,
        y_train,
        model_name,
        dataset_properties,
        model_kwargs,
        metric,
        sklearn_splitter: Any | None = None,
        splitting_kwargs: dict[str, Any] | None = None,
        additional_metrics: list[str] | None = None,
        X_valid=None,
        y_valid=None,
        X_test=None,
        y_test=None,
        store_preds: bool = False,
        preds_dir: Path | None = None,
    ) -> None:
        self.sklearn_splitter = sklearn_splitter
        self.splitting_kwargs = splitting_kwargs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.model_name = model_name
        self.dataset_properties = dataset_properties
        self.model_kwargs = model_kwargs
        self.metric = metric
        self.additional_metrics = [get_scorer(metric) for metric in additional_metrics] if additional_metrics else None
        self.store_preds = store_preds
        self.preds_dir = preds_dir

    def __call__(self, config: Configuration, seed: int | None = None) -> TargetFunctionResult:
        """
        Calls the target function. If a splitter is provided, the data is split into
        multiple splits and the model is trained on each split. Otherwise, the model
        is trained on a single split of the data.

        Args:
            config: Configuration to use for the model.
            seed: Seed to use for the splitter.

        Returns:
            TargetFunctionResult
        """
        if self.sklearn_splitter is None:
            return train_single_split(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                X_valid=self.X_valid,
                y_valid=self.y_valid,
                model_name=self.model_name,
                dataset_properties=self.dataset_properties,
                config=config,
                model_kwargs=self.model_kwargs,
                optimize_metric=self.metric,
                additional_metrics=self.additional_metrics,
                store_preds=self.store_preds,
                preds_dir=self.preds_dir,
            )
        else:
            return self._train_multiple_splits(config, seed)

    def _train_multiple_splits(self, config: Configuration, seed: int | None = None) -> TargetFunctionResult:
        """
        Trains a model on multiple splits of the data.

        Args:
            config: Configuration to use for the model.
            seed: Seed to use for the splitter.

        Returns:
            TargetFunctionResult
        """
        assert self.sklearn_splitter is not None, "Splitter must be provided to train on multiple splits."
        if self.splitting_kwargs is None:
            self.splitting_kwargs = {}
        splitter = self.sklearn_splitter(**self.splitting_kwargs)
        results = []
        for i, (train_index, valid_index) in enumerate(splitter.split(self.X_train, self.y_train)):
            print(f"Running split {i+1}/{self.splitting_kwargs.get('n_splits', 1)}")
            X_train, X_valid = self.X_train.iloc[train_index], self.X_train.iloc[valid_index]
            y_train, y_valid = self.y_train.iloc[train_index], self.y_train.iloc[valid_index]
            results.append(
                train_single_split(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    model_name=self.model_name,
                    dataset_properties=self.dataset_properties,
                    config=config,
                    model_kwargs=self.model_kwargs,
                    optimize_metric=self.metric,
                    additional_metrics=self.additional_metrics,
                    store_preds=self.store_preds,
                    preds_dir=self.preds_dir / f"split_{i}" if self.preds_dir is not None else None,
                )
            )

        return self._aggregate_results(results)

    def _aggregate_results(self, results: list[TargetFunctionResult]) -> TargetFunctionResult:
        """Aggregate results from multiple splits."""

        (train_metrics, valid_metrics, test_metrics) = self._get_aggregate_metrics_across_splits(results)

        info = TargetFunctionInfo(
            train_loss=sum(result.info["train_loss"] for result in results) / len(results),
            valid_loss=sum(result.info["valid_loss"] for result in results) / len(results),
            test_loss=sum(result.info["test_loss"] for result in results) / len(results),
            average_split_fit_time=sum(result.info["average_split_fit_time"] for result in results) / len(results),
            average_split_predict_time_test=sum(result.info["average_split_predict_time_test"] for result in results)
            / len(results),
            average_split_predict_time_valid=sum(result.info["average_split_predict_time_valid"] for result in results)
            / len(results),
            average_split_predict_time=sum(result.info["average_split_predict_time"] for result in results)
            / len(results),
            average_split_preprocess_time_train=sum(
                result.info["average_split_preprocess_time_train"] for result in results
            )
            / len(results),
            average_split_preprocess_time_valid=sum(
                result.info["average_split_preprocess_time_valid"] for result in results
            )
            / len(results),
            average_split_preprocess_time_test=sum(
                result.info["average_split_preprocess_time_test"] for result in results
            )
            / len(results),
            total_walltime=sum(result.info["total_walltime"] for result in results),
            config=results[0].info["config"],
            n_splits=len(results),
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
        )

        loss = sum(result.loss for result in results) / len(results)
        result = TargetFunctionResult(loss, info)
        return result

    @classmethod
    def _get_aggregate_metrics_across_splits(
        cls, results
    ) -> tuple[dict[str, float], dict[str, float] | None, dict[str, float] | None]:
        train_metrics: dict[str, float] = {}
        valid_metrics: dict[str, float] | None = {} if results[0].info["valid_metrics"] is not None else None
        test_metrics: dict[str, float] | None = {} if results[0].info["test_metrics"] is not None else None
        for _, result in enumerate(results):
            for metric_name, metric_value in result.info["train_metrics"].items():
                if metric_name in train_metrics:
                    train_metrics[metric_name] += metric_value
                else:
                    train_metrics[metric_name] = metric_value
            if valid_metrics is not None:
                for metric_name, metric_value in result.info["valid_metrics"].items():
                    if metric_name in valid_metrics:
                        valid_metrics[metric_name] += metric_value
                    else:
                        valid_metrics[metric_name] = metric_value
            if test_metrics is not None:
                for metric_name, metric_value in result.info["test_metrics"].items():
                    if metric_name in test_metrics:
                        test_metrics[metric_name] += metric_value
                    else:
                        test_metrics[metric_name] = metric_value

        for metric_name in train_metrics:
            train_metrics[metric_name] /= len(results)
        if valid_metrics is not None:
            for metric_name in valid_metrics:
                valid_metrics[metric_name] /= len(results)
        if test_metrics is not None:
            for metric_name in test_metrics:
                test_metrics[metric_name] /= len(results)

        return train_metrics, valid_metrics, test_metrics
