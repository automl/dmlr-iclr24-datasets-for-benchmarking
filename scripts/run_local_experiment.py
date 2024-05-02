"""run a experiment with a given config"""
import argparse
from sklearn.model_selection import StratifiedKFold
from tabular_data_experiments.metrics import get_scorer

from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.experiment_workflows.refit_experiment import run_refit_experiment
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.utils import add_common_args_to_parser, add_search_args_to_parser

parser = argparse.ArgumentParser(description="Run a experiment with a given config")
# parser.add_argument("--hpo", action="store_true", help="Whether to use HPO")
parser.add_argument(
    "--optimize", type=str, choices=["hpo", "default", "sobol"], help="Whether to use HPO", default="default"
)
# parser.add_argument("--hpo_config", type=Path, help="Path to configuration file of HPO")
parser.add_argument("--task_id", type=int, help="Task id", default=31)
parser.add_argument("--n_splits", type=int, help="Number of splits in HPO", default=5)
parser = add_common_args_to_parser(parser)
parser = add_search_args_to_parser(parser)

args = parser.parse_args()
if __name__ == "__main__":
    exp_dir = args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    experiment_seed = ExperimentSeed(seed)

    if args.run_config is not None:
        run_config = RunConfig.from_yaml(args.run_config)
    else:
        model_seed, data_seed, hpo_seed = (
            experiment_seed.get_seed("model"),
            experiment_seed.get_seed("data"),
            experiment_seed.get_seed("hpo"),
        )
        run_config = RunConfig(
            experiment_name=args.exp_name,
            seed=experiment_seed,
            manipulators=args.manipulators,
            metric=get_scorer(args.metric),
            model_name=args.model_name,
            fold_number=1,
            optimizer=args.optimize,
            optimizer_kwargs={
                # "seed": int(hpo_seed),
                # "output_directory": exp_dir / "hpo",
                # # "deterministic": True,
                # "name": args.exp_name,
                # "n_trials": args.n_trials,
                # "n_workers": args.n_workers,
                # "scenario_kwargs": {
                #     "seed": int(hpo_seed),
                #     "output_directory": exp_dir / "hpo",
                #     "deterministic": True,
                #     "name": args.exp_name,
                #     "n_trials": args.n_trials,
                #     "n_workers": args.n_workers,
                # }
            },
            model_kwargs={
                "seed": int(model_seed),
                "device": args.device,
            },  # , "budget": 15},  # , "iterations": 10},  # , "budget": 5},
            task_id=args.task_id,
            manipulator_kwargs={"remove_missing_values__threshold": 0.3, "remove_high_cardinality__threshold": 10},
            splitter=StratifiedKFold,
            splitter_kwargs={"n_splits": args.n_splits, "random_state": int(data_seed), "shuffle": True},
            additional_metrics=args.additional_metrics,
            store_preds=args.store_preds,
        )
    run_config.to_yaml(exp_dir / "run_config.yaml")
    result_config = run_refit_experiment(run_config, exp_dir)
    print(f"\n\nFinished with result: {result_config}")
