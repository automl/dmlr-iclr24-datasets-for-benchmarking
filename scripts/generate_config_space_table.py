import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from tabular_data_experiments.models.base_model import BaseModel

from tabular_data_experiments.results.constants import LABEL_NAMES
from tabular_data_experiments.run_result_config import DataclassEncoder
from tabular_data_experiments.utils.data_utils import get_required_dataset_info

argparser = argparse.ArgumentParser()
argparser.add_argument("--output_dir", type=Path, help="Output directory", default=Path("configspaces"))
argparser.add_argument("--seed", type=int, help="Seed", default=42)
argparser.add_argument("--include_models", type=str, nargs="+", help="Models to include", default=None)
args = argparser.parse_args()


if __name__ == "__main__":
    # HACK: Dummy dataset to get dataset properties which are
    # required for getting model configuration space
    dummy_dataset = {
        "X": pd.DataFrame(np.random.rand(10, 10)),
        "y": pd.Series(np.random.randint(0, 2, 10)),
        "categorical_indicator": [False] * 10,
    }
    dummy_dataset_properties = get_required_dataset_info(**dummy_dataset)

    for model in LABEL_NAMES.keys():
        if args.include_models is not None and model not in args.include_models:
            continue
        try:
            model_dir = args.output_dir
            model_dir.mkdir(parents=True, exist_ok=True)
            dummy_model = BaseModel.create(model, model_kwargs={}, dataset_properties=dummy_dataset_properties)
            # Beautify name
            model = model.replace("_early_stop_valid", "")
            # upper case first letter
            model = model[0].upper() + model[1:]
            # replace _ with space
            model = model.replace("_", " ")

            # add boilerplate latex table code
            table_header = """
\\begin{table}[ht]
\\centering
            """
            table = dummy_model.get_model_config_table()
            table_footer = f"""
\\caption{{Model configuration space for {model}.}}
\\label{{tab:{model.lower().replace(' ', '_')}_config_space}}
\\end{{table}}
            """
            table = table_header + table + table_footer
            # Write to txt file
            with open(model_dir / f"{model}.tex", "w") as f:
                f.write(table)
        except Exception as e:
            print(f"Could not generate configs for {model}")
            print(e)
            shutil.rmtree(model_dir)
            continue

    arguments = vars(args)
    json.dump(arguments, (args.output_dir / "arguments.json").open("w"), indent=4, cls=DataclassEncoder)
