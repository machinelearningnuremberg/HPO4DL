import random
import optuna
import ConfigSpace as CS
from datetime import datetime
from pathlib import Path
import os
import sys
import argparse
import json

objective_instance = None
checkpoint_root_path = None
checkpoint_map = {}
prev_result_map = {}
max_budget = 1000
current_budget = 0
max_epochs = 27
result_logger = None


# Define an objective function to be maximized.
def objective(trial: optuna.Trial):
    # Suggest values of the hyperparameters using a trial object.
    global checkpoint_root_path, checkpoint_map, max_budget, current_budget, max_epochs, result_logger

    configuration = {
        'lr': trial.suggest_float('lr', 1e-5, 1, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1, log=True),
        'model': trial.suggest_categorical('model', choices=["mobilevit_xxs", "dla60x_c", "edgenext_xx_small"]),
        'opt': trial.suggest_categorical('opt', choices=["sgd", "adam"]),
    }
    if configuration['opt'] == "sgd":
        configuration['momentum'] = trial.suggest_float('momentum', 0.1, 0.99)

    config_tuple = tuple(sorted(configuration.items()))
    if config_tuple not in checkpoint_map:
        new_checkpoint_path = checkpoint_root_path / f'trial_{len(checkpoint_map)}' / 'last'
        checkpoint_map[config_tuple] = (new_checkpoint_path, len(checkpoint_map))

    checkpoint_path, checkpoint_id = checkpoint_map[config_tuple]

    eval_result = []
    for epoch in range(1, max_epochs + 1):
        if (config_tuple, epoch) not in prev_result_map:
            eval_result = objective_instance.objective_function(
                configuration=configuration,
                epoch=epoch,
                previous_epoch=epoch - 1,
                checkpoint_path=checkpoint_path,
            )
            prev_result_map[(config_tuple, epoch)] = eval_result
        else:
            print(f"Configuration already evaluated. {config_tuple} epoch {epoch}")
            eval_result = prev_result_map[(config_tuple, epoch)]

        configuration_results = []
        for i, result in enumerate(eval_result):
            trial.report(result['metric'], result['epoch'])
            configuration_result = {
                **result,
                'configuration_id': checkpoint_id,
                'configuration': configuration,
            }
            configuration_results.append(configuration_result)
        result_logger.add_configuration_results(configuration_results)

        current_budget += 1
        if current_budget >= max_budget:
            result_logger.save_results()
            trial.study.stop()

        if trial.should_prune():
            raise optuna.TrialPruned()

    return eval_result[-1]['metric']


def main():
    parser = argparse.ArgumentParser(
        prog='hpo4dl',
        description='Hyperparameter optimization using optuna and timm libraries',
    )
    parser.add_argument('--output-dir', default='./optuna_results',
                        help='path to save results. (default: ./optuna_results)')
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
    parser.add_argument('--max-budget', type=int, default=1000,
                        help='maximum budget in epochs (default: 1000)')
    args = parser.parse_args()

    sys.argv = ['']

    from dummy_objective import DummyObjective
    from timm_objective import TimmObjective

    global checkpoint_root_path, result_logger, max_epochs, objective_instance, max_budget

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
    checkpoint_root_path = Path(storage_dir) / 'optuna' / experiment_name
    checkpoint_root_path.parent.mkdir(parents=True, exist_ok=True)

    result_root_path = Path(args.output_dir) / experiment_name
    result_logger = ResultLogger(path=result_root_path, minimize=False)

    with open(result_root_path / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Create a study object and optimize the objective function.
    max_epochs = args.max_epochs
    max_budget = args.max_budget
    study = optuna.create_study(
        study_name=f'hpo4dl_{args.seed}',
        direction='maximize',
    )
    study.optimize(objective)

    # print("Best Configuration Info", best_configuration)


if __name__ == "__main__":
    main()
