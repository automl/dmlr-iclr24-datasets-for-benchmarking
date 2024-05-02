"""Create skorch classifiers from neural network modules"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring, LRScheduler
from skorch.dataset import Dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tabular_data_experiments.models.model_library.tabular.models.ft_transformer import (
    Transformer,
)
from tabular_data_experiments.models.model_library.tabular.models.mlp import MLP
from tabular_data_experiments.models.model_library.tabular.models.resnet import ResNet


class NeuralNetworkSkorch(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        if self.criterion_.__class__.__name__ == "BCEWithLogitsLoss":
            y_true = y_true.float()
        return super().get_loss(y_pred, y_true, *args, **kwargs)


MODEL_TYPES = {
    "mlp": MLP,
    "resnet": ResNet,
    "transformer": Transformer,
}


def train_split(X, y, X_val, y_val):
    # need to return train as well,
    # otherwise issue with epoch scoring
    # for early stopping
    train_dataset = Dataset(X, y)
    valid_dataset = Dataset(X_val, y_val)
    return train_dataset, valid_dataset


def create_skorch_model(
    config,
    model_type: str,
    id=0,
    use_checkpoints: bool = True,
    device: str = "cpu",
    checkpoint_dir: str | None = None,
    categorical_indicator: Iterable[bool] | None = None,
    categories: Iterable[int] | None = None,
    output_shape: int = 1,
    num_numerical_features: int = 0,
    n_classes: int = 2,
    es_patience: int = 40,
    lr_patience: int = 30,
    verbose: int = 0,
    max_epochs: int = 100,
):
    config = config.copy()

    lr_scheduler = config.pop("lr_scheduler", False)

    optimizer = config.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")

    callbacks = [
        EarlyStopping(monitor="valid_loss", patience=es_patience),
    ]  # TODO try with train_loss, and in this case use checkpoint
    callbacks.append(EpochScoring(scoring="accuracy", name="train_accuracy", on_train=True, use_caching=False))
    callbacks.append(EpochScoring(scoring="accuracy", name="train_loss", on_train=True, use_caching=False))

    if lr_scheduler:
        callbacks.append(
            LRScheduler(policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2)
        )  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(
            Checkpoint(dirname=checkpoint_dir, f_params=r"params_{}.pt".format(id), f_optimizer=None, f_criterion=None)
        )

    if n_classes == 2:
        criterion = BCEWithLogitsLoss()
    else:
        criterion = CrossEntropyLoss()

    if categorical_indicator is not None and not any(categorical_indicator):
        categories = None

    skorch_model = NeuralNetworkSkorch(
        MODEL_TYPES[model_type],
        optimizer=optimizer,
        # Shuffle training data on each epoch
        iterator_train__shuffle=True,
        module__d_numerical=num_numerical_features,
        module__categories=categories,
        module__d_out=output_shape,
        module__regression=False,
        module__categorical_indicator=categorical_indicator,
        verbose=verbose,
        callbacks=callbacks,
        predict_nonlinearity="auto",
        device=device,
        max_epochs=max_epochs,
        criterion=criterion,
        train_split=train_split,
        # Since we have batch norm which can not be evaluated on
        # a batch of size 1, no problem for predicting though,
        # since batch norm is not used for inference.
        iterator_train__drop_last=True,
        **config,
    )

    return skorch_model
