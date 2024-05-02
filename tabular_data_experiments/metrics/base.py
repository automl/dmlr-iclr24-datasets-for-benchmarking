""" Base class for metrics
Code adapted from https://github.com/automl/auto-sklearn/blob/development/autosklearn/metrics/__init__.py

"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional

import os

import numpy as np
import sklearn
from sklearn.utils.multiclass import type_of_target


class Scorer(object, metaclass=ABCMeta):
    def __init__(
        self,
        name: str,
        score_func: Callable,
        optimum: float,
        worst_possible_result: float,
        sign: float,
        kwargs: Any,
    ) -> None:
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._worst_possible_result = worst_possible_result
        self._sign = sign

    @abstractmethod
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        pass

    def __repr__(self) -> str:
        return self.name


class _PredictScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        type_true = type_of_target(y_true)
        if (
            type_true == "binary"
            and type_of_target(y_pred) == "continuous"
            and (len(y_pred.shape) == 1 or y_pred.shape[1] == 1)
        ):
            # For a pred scorer, no threshold, nor probability is required
            # If y_true is binary, and y_pred is continuous
            # it means that a rounding is necessary to obtain the binary class
            y_pred = np.around(y_pred, decimals=0)
        elif len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or type_true == "continuous":
            # must be regression, all other task types would return at least
            # two probabilities
            pass
        elif type_true in ["binary", "multiclass"]:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == "multilabel-indicator":
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        elif type_true == "continuous-multioutput":
            pass
        else:
            raise ValueError(type_true)

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight
        return self._sign * self._score_func(y_true, y_pred, **scorer_kwargs, **self._kwargs)


class _ProbaScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        X_data : array-like [n_samples x n_features]
            X data used to obtain the predictions: each row x_j corresponds to the input
             used to obtain predictions y_j
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        n_classes = np.unique(y_true).shape[0]
        if self._score_func is sklearn.metrics.log_loss or self._score_func is sklearn.metrics.roc_auc_score:
            # roc auc score also requires labels argument. To recreate
            # failure run 361589, split 0, fold 0, where in valid set label 19 is missing.
            n_labels_pred = np.array(y_pred).reshape((len(y_pred), -1)).shape[1]
            n_labels_test = len(np.unique(y_true))
            is_1d_pred_proba = n_labels_pred == 1 and n_classes == 2
            if (
                n_labels_pred != n_labels_test
                and not is_1d_pred_proba  # happens for deep models, since they use sigmoid
            ):
                if self.name == "roc_auc_ovr" and os.environ.get("OPENML_TASK_ID", 1) in [
                    "361589",
                    "361625",
                    "361661"
                ]:  # since we want to handle it differently
                    if n_labels_pred > n_labels_test:
                        # Drop the extra columns in y_pred
                        y_pred = y_pred[:, np.unique(y_true)]
                        # rescale probabilities
                        y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
                    else:
                        # Add extra columns to y_pred
                        y_pred = np.hstack(
                            [
                                y_pred,
                                np.zeros((y_pred.shape[0], n_labels_test - n_labels_pred)),
                            ]
                        )
                else:
                    labels = list(range(n_labels_pred))
                    if sample_weight is not None:
                        return self._sign * self._score_func(
                            y_true,
                            y_pred,
                            sample_weight=sample_weight,
                            labels=labels,
                            **self._kwargs,
                        )
                    else:
                        return self._sign * self._score_func(y_true, y_pred, labels=labels, **self._kwargs)

        if self._score_func is sklearn.metrics.roc_auc_score and n_classes == 2:
            # happens for deep models
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                y_pred = y_pred[:, 0]
            # happens for other models
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight
        return self._sign * self._score_func(y_true, y_pred, **scorer_kwargs, **self._kwargs)


class _ThresholdScorer(Scorer):
    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        sample_weight: Optional[List[float]] = None,
    ) -> float:
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        X_data : array-like [n_samples x n_features]
            X data used to obtain the predictions: each row x_j corresponds to the input
             used to obtain predictions y_j
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            if y_pred.ndim > 1 and y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        scorer_kwargs = {}  # type: Dict[str, Union[List[float], np.ndarray]]
        if sample_weight is not None:
            scorer_kwargs["sample_weight"] = sample_weight

        return self._sign * self._score_func(y_true, y_pred, **scorer_kwargs, **self._kwargs)


def make_scorer(
    name: str,
    score_func: Callable,
    *,
    optimum: float = 1.0,
    worst_possible_result: float = 0.0,
    greater_is_better: bool = True,
    needs_proba: bool = False,
    needs_threshold: bool = False,
    **kwargs: Any,
) -> Scorer:
    """Make a scorer from a performance metric or loss function.
    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.
    Parameters
    ----------
    name: str
        Descriptive name of the metric
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.
    worst_possible_result : int of float, default=0
        The worst score achievable by the score function, i.e. minimum in case of
        scorer function and maximum in case of loss function.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score; greater is better or set
        greater_is_better to False.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba and needs_threshold:
        raise ValueError("Set either needs_proba or needs_threshold to True, but not both.")

    cls = None  # type: Any
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(name, score_func, optimum, worst_possible_result, sign, kwargs)
