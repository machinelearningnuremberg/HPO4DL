import random
import optuna
import ConfigSpace as CS
from datetime import datetime
from pathlib import Path
import os

from dummy_objective import DummyObjective
from timm_objective import objective_function
from hpo4dl.utils.result_logger import ResultLogger

checkpoint_root_path = None
checkpoint_map = {}
prev_result_map = {}
max_total_budget = 1000
current_budget = 0
max_epochs = 27
result_logger: ResultLogger = None


# Define an objective function to be maximized.
def objective(trial: optuna.Trial):
    # Suggest values of the hyperparameters using a trial object.
    global checkpoint_root_path, checkpoint_map, max_total_budget, current_budget, max_epochs, result_logger

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
        new_checkpoint_path = checkpoint_root_path / f'trial_{len(checkpoint_map)}' / 'last.pth.tar'
        checkpoint_map[config_tuple] = (new_checkpoint_path, len(checkpoint_map))

    checkpoint_path, checkpoint_id = checkpoint_map[config_tuple]

    eval_result = []
    for epoch in range(1, max_epochs + 1):
        if (config_tuple, epoch) not in prev_result_map:
            eval_result = objective_function(
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
        if current_budget >= max_total_budget:
            result_logger.save_results()
            trial.study.stop()

        if trial.should_prune():
            raise optuna.TrialPruned()

    return eval_result[-1]['metric']


def main():
    seed = 0

    random.seed(seed)

    global checkpoint_root_path, result_logger, max_epochs
    experiment_name = f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_root_path = Path(os.path.expanduser('~/hpo4dl/optuna')) / experiment_name
    checkpoint_root_path.parent.mkdir(parents=True, exist_ok=True)

    result_root_path = Path('./optuna_results') / experiment_name
    result_logger = ResultLogger(path=result_root_path, minimize=False)

    # Create a study object and optimize the objective function.
    max_epochs = 27
    study = optuna.create_study(
        direction='maximize',
    )
    study.optimize(objective)

    # print("Best Configuration Info", best_configuration)


if __name__ == "__main__":
    main()
