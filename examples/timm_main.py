from pathlib import Path
import random
import ConfigSpace as CS
import timm

from dummy_objective import DummyObjective
from timm_objective import objective_function
from hpo4dl.tuner import Tuner


def main():
    seed = 0
    # models = timm.list_models()
    # print(models)
    random.seed(seed)
    config_space = CS.ConfigurationSpace(seed=seed)
    # config_space.add_hyperparameters([
    #     # CS.UniformIntegerHyperparameter('batch_size', lower=16, upper=512, log=True),
    #     CS.UniformFloatHyperparameter('lr', lower=1e-4, upper=0.1, log=True),
    #     CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=0.1, log=False),
    #     CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99, log=True),
    # ])
    config_space.add_hyperparameters([
        CS.UniformFloatHyperparameter('lr', lower=5e-5, upper=5, log=True),
        CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=5, log=True),
        # CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99, log=True),
    ])

    tuner = Tuner(
        objective_function=objective_function,
        configuration_space=config_space,
        minimize=False,
        max_total_budget=None,
        optimizer='hyperband',
        seed=seed,
        max_epochs=81,
        result_path='./result',
    )
    best_config = tuner.run()
    print(best_config)


if __name__ == "__main__":
    main()
