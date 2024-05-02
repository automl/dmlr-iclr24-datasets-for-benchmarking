from __future__ import annotations

from tabular_data_experiments.data_manipulation.manipulators import (
    DataManipulaterType,
    balance_binarize,
    remove_high_cardinality,
    remove_missing_values,
    remove_pseudo_categorical,
    remove_rows_with_missing_values,
    subsample_columns,
    subsample_rows,
)

DataManipulators: dict[str, DataManipulaterType] = {
    "remove_missing_values": remove_missing_values,
    "remove_pseudo_categorical": remove_pseudo_categorical,
    "remove_high_cardinality": remove_high_cardinality,
    "remove_rows_with_missing_values": remove_rows_with_missing_values,
    "balance_binarize": balance_binarize,
    "subsample_rows": subsample_rows,
    "subsample_columns": subsample_columns,
}
