import random
import ConfigSpace as CS
from datetime import datetime
from pathlib import Path
import os
import sys
from smac import MultiFidelityFacade as MFFacade
from smac import Scenario, RunHistory
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

from dummy_objective import DummyObjective
from timm_objective import objective_function

checkpoint_root_path = None
checkpoint_map = {}
prev_result_map = {}
prev_epoch_map = {}
max_total_budget = 1000
current_budget = 0
max_epochs = 27
result_logger = None


# Define an objective function to be maximized.
def objective(config: CS.Configuration, seed: int = 0, budget: int = 27):
    # Suggest values of the hyperparameters using a trial object.
    global checkpoint_root_path, checkpoint_map, max_total_budget, current_budget, max_epochs, \
        result_logger, prev_result_map, prev_epoch_map

    # configuration = {
    #     'lr': config.get("lr", 1e-3),
    #     'weight_decay': config.get("weight_decay", 1e-4),
    # }
    configuration = dict(config)

    config_tuple = tuple(sorted(configuration.items()))
    if config_tuple not in checkpoint_map:
        new_checkpoint_path = checkpoint_root_path / f'trial_{len(checkpoint_map)}' / 'last.pth.tar'
        checkpoint_map[config_tuple] = (new_checkpoint_path, len(checkpoint_map))

    checkpoint_path, checkpoint_id = checkpoint_map[config_tuple]

    epoch = int(budget)
    if (config_tuple, epoch) not in prev_result_map:
        previous_epoch = prev_epoch_map[config_tuple] if config_tuple in prev_epoch_map else 0
        eval_result = objective_function(
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

    configuration_results = []
    for i, result in enumerate(eval_result):
        configuration_result = {
            **result,
            'configuration_id': checkpoint_id,
            'configuration': configuration,
        }
        configuration_results.append(configuration_result)
    result_logger.add_configuration_results(configuration_results)

    metrics = [v['metric'] for v in eval_result]
    return 1 - max(metrics)


def main():
    seed = 0

    random.seed(seed)

    global checkpoint_root_path, result_logger, max_epochs

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    sys.path.append(parent_dir)
    from hpo4dl.utils.result_logger import ResultLogger

    experiment_name = f'experiment_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    checkpoint_root_path = Path(os.path.expanduser('~/hpo4dl/optuna')) / experiment_name
    checkpoint_root_path.parent.mkdir(parents=True, exist_ok=True)

    result_root_path = Path('./smac_results') / experiment_name
    result_logger = ResultLogger(path=result_root_path, minimize=False)

    config_space = CS.ConfigurationSpace(seed=seed)
    config_space.add_hyperparameters([
        CS.UniformFloatHyperparameter('lr', lower=1e-5, upper=1, log=True),
        CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1, log=True),
        CS.CategoricalHyperparameter('model', choices=["mobilevit_xxs", "dla60x_c", "edgenext_xx_small"]),
        CS.CategoricalHyperparameter('opt', choices=["sgd", "adam"]),
        CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99),
    ])
    cond = CS.EqualsCondition(config_space['momentum'], config_space['opt'], "sgd")
    config_space.add_condition(cond)

    # Create a study object and optimize the objective function.
    max_epochs = 27
    scenario = Scenario(
        config_space,
        deterministic=True,
        min_budget=1,
        max_budget=max_epochs,
        n_trials=225
    )
    intensifier = Hyperband(scenario, incumbent_selection="highest_observed_budget")
    smac = MFFacade(
        scenario,
        objective,
        intensifier=intensifier,
        overwrite=True,
    )
    best_configuration = smac.optimize()

    print("Best Configuration Info", best_configuration)

    incumbent_cost = smac.validate(best_configuration)
    print(f"Incumbent cost: {incumbent_cost}")


if __name__ == "__main__":
    main()
