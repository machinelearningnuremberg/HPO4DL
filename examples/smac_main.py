import random
import ConfigSpace as CS
from datetime import datetime
from pathlib import Path
import os
import sys
import time
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario, RunHistory
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving
import argparse
import json

objective_instance = None
checkpoint_root_path = None
checkpoint_map = {}
prev_result_map = {}
prev_epoch_map = {}
current_budget = 0
result_logger = None
surrogate_overhead_start_time = 0


# Define an objective function to be maximized.
def objective(config: CS.Configuration, seed: int = 0, budget: int = 27):
    # Suggest values of the hyperparameters using a trial object.
    global checkpoint_root_path, checkpoint_map, result_logger, prev_result_map, prev_epoch_map, \
        surrogate_overhead_start_time

    surrogate_overhead_time = time.perf_counter() - surrogate_overhead_start_time

    configuration = dict(config)

    config_tuple = tuple(sorted(configuration.items()))
    if config_tuple not in checkpoint_map:
        new_checkpoint_path = checkpoint_root_path / f'trial_{len(checkpoint_map)}' / 'last'
        checkpoint_map[config_tuple] = (new_checkpoint_path, len(checkpoint_map))

    checkpoint_path, checkpoint_id = checkpoint_map[config_tuple]

    epoch = int(budget)
    model_start_time = time.perf_counter()
    if (config_tuple, epoch) not in prev_result_map:
        previous_epoch = prev_epoch_map[config_tuple] if config_tuple in prev_epoch_map else 0
        eval_result = objective_instance.objective_function(
            configuration=configuration,
            epoch=epoch,
            previous_epoch=previous_epoch,
            checkpoint_path=checkpoint_path,
        )
        prev_result_map[(config_tuple, epoch)] = eval_result
        prev_epoch_map[config_tuple] = epoch
    else:
        print(f"Configuration already evaluated. {config_tuple} epoch {epoch}")
        eval_result = prev_result_map[(config_tuple, epoch)]

    model_end_time = time.perf_counter()
    model_execution_time = model_end_time - model_start_time
    single_model_execution_time = model_execution_time / len(eval_result)

    single_surrogate_overhead_time = surrogate_overhead_time / len(eval_result)

    configuration_results = []
    for i, result in enumerate(eval_result):
        configuration_result = {
            **result,
            'configuration_id': checkpoint_id,
            'configuration': configuration,
            'time': single_model_execution_time,
            'overhead_time': single_surrogate_overhead_time,
        }
        configuration_results.append(configuration_result)
    result_logger.add_configuration_results(configuration_results)

    metrics = [v['metric'] for v in eval_result]
    surrogate_overhead_start_time = time.perf_counter()

    return 1 - max(metrics)


def main():
    parser = argparse.ArgumentParser(
        prog='hpo4dl',
        description='Hyperparameter optimization using smac and timm libraries',
    )
    parser.add_argument('--output-dir', default='./smac_results',
                        help='path to save results. (default: ./smac_results)')
    parser.add_argument('--storage-dir', default='~/hpo4dl',
                        help='path to temporary checkpoint storage directory. (default: ~/hpo4dl)')
    parser.add_argument('--dataset', type=str, default='torch/cifar10',
                        help='dataset name (default: torch/cifar10)')
    parser.add_argument('--train-split', type=str, default='train',
                        help='train split name (default: train)')
    parser.add_argument('--val-split', type=str, default='validation',
                        help='validation split name (default: validation)')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--max-epochs', type=int, default=27,
                        help='maximum epochs (default: 27)')
    parser.add_argument('--max-budget', type=int, default=225,
                        help='maximum budget in epochs (default: 225)')
    args = parser.parse_args()

    sys.argv = ['']

    from dummy_objective import DummyObjective
    from timm_objective import TimmObjective

    global checkpoint_root_path, result_logger, objective_instance, surrogate_overhead_start_time

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    sys.path.append(parent_dir)
    from hpo4dl.utils.result_logger import ResultLogger

    storage_dir = os.path.expanduser(args.storage_dir)
    objective_instance = TimmObjective(
        dataset=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        storage_dir=storage_dir,
    )
    # objective_instance = DummyObjective(seed=args.seed)

    experiment_name = f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_root_path = Path(storage_dir) / 'smac' / experiment_name
    checkpoint_root_path.parent.mkdir(parents=True, exist_ok=True)

    result_root_path = Path(args.output_dir) / experiment_name
    result_logger = ResultLogger(path=result_root_path, minimize=False)

    with open(result_root_path / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Create a study object and optimize the objective function.
    scenario = Scenario(
        objective_instance.configspace,
        deterministic=True,
        min_budget=1,
        max_budget=args.max_epochs,
        n_trials=args.max_budget
    )
    intensifier = Hyperband(scenario)
    smac = MFFacade(
        scenario,
        objective,
        intensifier=intensifier,
        overwrite=True,
    )

    surrogate_overhead_start_time = time.perf_counter()

    incumbent = smac.optimize()

    print("incumbent Info", incumbent)

    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")


if __name__ == "__main__":
    main()
