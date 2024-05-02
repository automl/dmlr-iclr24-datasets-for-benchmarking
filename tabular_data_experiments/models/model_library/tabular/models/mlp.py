import typing as ty

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        *,
        d_numerical: int,
        n_layers: int,
        d_layers: int,  # CHANGED
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        regression: bool,
        categorical_indicator,
    ) -> None:
        super().__init__()

        self.regression = regression
        self.categorical_indicator = torch.as_tensor(categorical_indicator, dtype=torch.bool)  # Added

        d_in = d_numerical
        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f"{self.category_embeddings.weight.shape=}")

        d_layers = [d_layers for _ in range(n_layers)]  # CHANGED

        self.layers = nn.ModuleList([nn.Linear(d_layers[i - 1] if i else d_in, x) for i, x in enumerate(d_layers)])
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x, **kwargs):
        if not isinstance(x, torch.Tensor):
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

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x
