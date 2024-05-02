"""Metrics for tabular data experiments.
Code adapted from https://github.com/automl/auto-sklearn/blob/development/autosklearn/metrics/__init__.py
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import sklearn.metrics

from tabular_data_experiments.metrics.base import Scorer, make_scorer

MAXINT = 2**31 - 1

# Standard Classification Scores
accuracy = make_scorer("accuracy", sklearn.metrics.accuracy_score)
balanced_accuracy = make_scorer("balanced_accuracy", sklearn.metrics.balanced_accuracy_score)

# Score functions that need decision values
roc_auc = make_scorer(
    "roc_auc",
    sklearn.metrics.roc_auc_score,
    greater_is_better=True,
    needs_proba=True,
    multi_class="ovo",
)
average_precision = make_scorer("average_precision", sklearn.metrics.average_precision_score, needs_threshold=True)

brier_score = make_scorer(
    "brier_score",
    sklearn.metrics.brier_score_loss,
    optimum=0,
    needs_threshold=True,
    worst_possible_result=1,
    greater_is_better=False,
)
f1 = make_scorer("f1", sklearn.metrics.f1_score, average="binary", zero_division=0)
f1_micro = make_scorer(
    "f1_micro",
    sklearn.metrics.f1_score,
    average="micro",
    zero_division=0,
)
f1_macro = make_scorer(
    "f1_macro",
    sklearn.metrics.f1_score,
    average="macro",
    zero_division=0,
)
f1_weighted = make_scorer(
    "f1_weighted",
    sklearn.metrics.f1_score,
    average="weighted",
    zero_division=0,
)
f1_samples = make_scorer(
    "f1_samples",
    sklearn.metrics.f1_score,
    average="samples",
    zero_division=0,
)
jaccard = make_scorer(
    "jaccard",
    sklearn.metrics.jaccard_score,
    zero_division=0,
)
jaccard_micro = make_scorer(
    "jaccard_micro",
    sklearn.metrics.jaccard_score,
    average="micro",
    zero_division=0,
)
jaccard_macro = make_scorer(
    "jaccard_macro",
    sklearn.metrics.jaccard_score,
    average="macro",
    zero_division=0,
)
jaccard_weighted = make_scorer(
    "jaccard_weighted",
    sklearn.metrics.jaccard_score,
    average="weighted",
    zero_division=0,
)
jaccard_samples = make_scorer(
    "jaccard_samples",
    sklearn.metrics.jaccard_score,
    average="samples",
    zero_division=0,
)
roc_auc_ovr = make_scorer(
    "roc_auc_ovr",
    sklearn.metrics.roc_auc_score,
    multi_class="ovr",
    greater_is_better=True,
    needs_proba=True,
)
zero_one_loss = make_scorer(
    "zero_one_loss",
    sklearn.metrics.zero_one_loss,
    greater_is_better=False,
    optimum=0,
    worst_possible_result=1,
)
# NOTE: zero_division
#
#   Specified as the explicit default, see sklearn docs:
#   https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn-metrics-precision-score
precision = make_scorer("precision", sklearn.metrics.precision_score, zero_division=0)
recall = make_scorer("recall", sklearn.metrics.recall_score, zero_division=0)


# Score function for probabilistic classification
log_loss = make_scorer(
    "log_loss",
    sklearn.metrics.log_loss,
    optimum=0,
    worst_possible_result=MAXINT,
    greater_is_better=False,
    needs_proba=True,
)


def calculate_scores(
    solution: np.ndarray,
    prediction: np.ndarray,
    metrics: Sequence[Scorer],
) -> Dict[str, float]:
    """
    Returns the scores (a magnitude that allows casting the
    optimization problem as a maximization one) for the
    given Scorer objects.
    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metrics: Sequence[Scorer]
        A list of objects that hosts a function to calculate how good the
        prediction is according to the solution.
    Returns
    -------
    Dict[str, float]
    """
    to_score = list(metrics)
    score_dict = dict()

    for metric_ in to_score:
        try:
            score_dict[metric_.name] = _compute_single_scorer(
                metric=metric_,
                prediction=prediction,
                solution=solution,
            )
        except ValueError as e:
            if e.args[0] == "multiclass format is not supported":
                continue
            elif e.args[0] == "Samplewise metrics are not available " "outside of multilabel classification.":
                continue
            elif (
                e.args[0] == "Target is multiclass but "
                "average='binary'. Please choose another average "
                "setting, one of [None, 'micro', 'macro', 'weighted']."
            ):
                continue
            else:
                raise e

    return score_dict


def calculate_loss(
    solution: np.ndarray,
    prediction: np.ndarray,
    metric: Scorer,
) -> float:
    """Calculate the loss with a given metric
    Parameters
    ----------
    solution: np.ndarray
        The solutions
    prediction: np.ndarray
        The predictions generated
    task_type: int
        The task type of the problem
    metric: Scorer
        The metric to use
    X_data: Optional[SUPPORTED_XDATA_TYPES]
        X data used to obtain the predictions
    """
    losses = calculate_losses(
        solution=solution,
        prediction=prediction,
        metrics=[metric],
    )
    return losses[metric.name]


def calculate_losses(
    solution: np.ndarray,
    prediction: np.ndarray,
    metrics: Sequence[Scorer],
) -> Dict[str, float]:
    """
    Returns the losses (a magnitude that allows casting the
    optimization problem as a minimization one) for the
    given Scorer objects.
    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metrics: Sequence[Scorer]
        A list of objects that hosts a function to calculate how good the
        prediction is according to the solution.
    X_data: Optional[SUPPORTED_XDATA_TYPES]
        X data used to obtain the predictions
    scoring_functions: List[Scorer]
        A list of metrics to calculate multiple losses
    Returns
    -------
    Dict[str, float]
        A loss function for each of the provided scorer objects
    """
    score = calculate_scores(
        solution=solution,
        prediction=prediction,
        metrics=metrics,
    )

    # we expect a dict() object for which we should calculate the loss
    loss_dict = dict()
    for metric_ in list(metrics):
        if metric_.name not in score:
            continue
        loss_dict[metric_.name] = metric_._optimum - score[metric_.name]
    return loss_dict


def _compute_single_scorer(
    metric: Scorer,
    prediction: np.ndarray,
    solution: np.ndarray,
) -> float:
    """
    Returns a score (a magnitude that allows casting the
    optimization problem as a maximization one) for the
    given Scorer object
    Parameters
    ----------
    solution: np.ndarray
        The ground truth of the targets
    prediction: np.ndarray
        The best estimate from the model, of the given targets
    task_type: int
        To understand if the problem task is classification
        or regression
    metric: Scorer
        Object that host a function to calculate how good the
        prediction is according to the solution.
    X_data : array-like [n_samples x n_features]
        X data used to obtain the predictions
    Returns
    -------
    float
    """
    score = metric(solution, prediction)
    return score


SCORERS_DICT: dict[str, Scorer] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "roc_auc": roc_auc,
    "log_loss": log_loss,
    "average_precision": average_precision,
    "brier_score": brier_score,
    "f1": f1,
    "f1_micro": f1_micro,
    "f1_macro": f1_macro,
    "f1_weighted": f1_weighted,
    "f1_samples": f1_samples,
    "jaccard": jaccard,
    "jaccard_micro": jaccard_micro,
    "jaccard_macro": jaccard_macro,
    "jaccard_weighted": jaccard_weighted,
    "jaccard_samples": jaccard_samples,
    "roc_auc_ovr": roc_auc_ovr,
    "zero_one_loss": zero_one_loss,
    "precision": precision,
    "recall": recall,
}

BINARY_METRICS = ["average_precision", "precision", "recall", "brier_score", "f1", "jaccard"]


def get_scorer(metric_name: str) -> Scorer:
    """Get the metric."""
    if metric_name not in SCORERS_DICT:
        raise ValueError(f"Unknown metric: {metric_name}")
    return SCORERS_DICT[metric_name]
