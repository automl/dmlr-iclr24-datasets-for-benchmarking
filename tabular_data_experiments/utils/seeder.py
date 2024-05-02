"""Experiment seed."""
from __future__ import annotations

import numpy as np


class ExperimentSeed:
    """Experiment seed."""

    def __init__(self, seed: int) -> None:
        """Initialize the experiment seed."""
        self.seed = seed
        self.random_state: np.random.Generator = np.random.default_rng(seed)
        self.generated_seeds: dict[str, int] = {}

    def get_seed(self, name: str) -> int:
        """Get the seed for the given name."""
        if name not in self.generated_seeds:
            self.generated_seeds[name] = int(self.random_state.integers(0, 2**32))
        return self.generated_seeds[name]

    def to_json(self) -> dict[str, int | dict[str, int]]:
        """Get the json representation."""
        return {
            "initial_seed": self.seed,
            "generated_seeds": {k: int(v) for k, v in self.generated_seeds.items()},
        }

    def __repr__(self) -> str:
        """Get the string representation."""
        return f"ExperimentSeed(seed={self.seed})"
