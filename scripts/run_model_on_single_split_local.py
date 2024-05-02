"""run a experiment with a given config"""
import argparse
from pathlib import Path


from tabular_data_experiments.experiment_workflows.fit_single_split import (
    run_single_split_experiment,
)
from tabular_data_experiments.metrics import get_scorer
from tabular_data_experiments.run_result_config import RunConfig
from tabular_data_experiments.utils.seeder import ExperimentSeed
from tabular_data_experiments.utils.utils import add_args_to_parser, get_split_experiment_dir

parser = argparse.ArgumentParser(description="Run a experiment with a given config")
arguments = [
    "--fold_number",
    "--split_number",
    "--task_id",
    "--n_splits",
    "--run_config",
    "--n_configs",
    "--exp_dir",
    "--seed",
    "--manipulators",
    "--model_name",
    "--metric",
    "--data_loader",
    "--exp_name",
    "--device",
    "--additional_metrics",
    "--store_preds",
    "--study",
    "--custom_study",
    "--overwrite",
    "--config_id",
    
]
parser = add_args_to_parser(parser, arguments)

args = parser.parse_args()

if __name__ == "__main__":
    exp_dir = args.exp_dir
    exp_dir = get_split_experiment_dir(
        exp_dir=exp_dir,
        exp_name=args.exp_name,
        task_id=args.task_id,
        fold_number=args.fold_number,
        split_number=args.split_number,
        config_id=args.config_id,
    )
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
            fold_number=args.fold_number,
            split_id=args.split_number,
            config_id=args.config_id,
            model_kwargs={
                "seed": int(model_seed),
                "device": args.device,
            },
            task_id=args.task_id,
            additional_metrics=args.additional_metrics,
            store_preds=args.store_preds,
        )
    run_config.to_yaml(exp_dir / "run_config.yaml")
    result_config = run_single_split_experiment(run_config, exp_dir)
    print(f"\n\nFinished with result: {result_config}")
