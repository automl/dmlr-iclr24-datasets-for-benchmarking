import typing as ty

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tabular_data_experiments.models.model_library.tabular.utils import (
    get_activation_fn,
    get_nonglu_activation_fn,
)


class ResNet(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        d: int,
        d_hidden_factor: float,
        n_layers: int,
        activation: str,
        normalization: str,
        hidden_dropout: float,
        residual_dropout: float,
        d_out: int,
        regression: bool,
        categorical_indicator,
    ) -> None:
        super().__init__()
        # categories = None #TODO
        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[normalization](d)

        self.categorical_indicator = torch.as_tensor(categorical_indicator, dtype=torch.bool)  # Added
        self.regression = regression
        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(d, d_hidden * (2 if activation.endswith("glu") else 1)),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x, **kwargs) -> Tensor:
        if not isinstance(x, Tensor):
            x = x[0]  # HACK: skorch passes a tuple of X, y to forward
        if self.categorical_indicator is not None and self.categorical_indicator.any().item():
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()  # TODO
        else:
            x_num = x
            x_cat = None
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(self.category_embeddings(x_cat + self.category_offsets[None]).view(x_cat.size(0), -1))
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer["linear1"](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x
