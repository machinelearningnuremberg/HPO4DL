import sys
import os
import random
import ConfigSpace as CS

from dummy_objective import DummyObjective
from timm_objective import objective_function


def main():
    seed = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    sys.path.append(parent_dir)
    from hpo4dl.tuner import Tuner

    random.seed(seed)
    config_space = CS.ConfigurationSpace(seed=seed)
    config_space.add_hyperparameters([
        CS.UniformFloatHyperparameter('lr', lower=5e-5, upper=5, log=True),
        CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=5, log=True),
        # CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99, log=True),
    ])

    tuner = Tuner(
        objective_function=objective_function,
        # objective_function=DummyObjective.dummy_objective_function,
        configuration_space=config_space,
        minimize=False,
        max_total_budget=1000,
        optimizer='dyhpo',
        seed=seed,
        max_epochs=27,
        result_path='./hpo4dl_results',
    )
    best_configuration = tuner.run()
    print("Best Configuration Info", best_configuration)


if __name__ == "__main__":
    main()
