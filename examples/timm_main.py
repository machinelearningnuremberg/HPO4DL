import sys
import os
import random
import ConfigSpace as CS

from dummy_objective import DummyObjective
from timm_objective import objective_function


def main():
    seed = 0

    # from pathlib import Path
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # files = ['dyhpo', 'hyperband']
    # root_path = Path('./hpo4dl_results')
    # result_data = {}
    # metrics_data = pd.DataFrame()
    # for f in files:
    #     source_path = root_path / f / "hpo4dl_results.csv"
    #     result_data[f] = pd.read_csv(source_path)
    #     metrics_data[f] = result_data[f]['best_metric']
    # sns.lineplot(metrics_data)
    # plt.savefig("comparison.png", dpi=200)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    print(parent_dir)
    sys.path.append(parent_dir)
    from hpo4dl.tuner import Tuner

    random.seed(seed)
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

    tuner = Tuner(
        # objective_function=objective_function,
        objective_function=DummyObjective.dummy_objective_function,
        configuration_space=config_space,
        minimize=False,
        max_total_budget=1000,
        optimizer='hyperband',
        seed=seed,
        max_epochs=27,
        result_path='./hpo4dl_results',
    )
    best_configuration = tuner.run()
    print("Best Configuration Info", best_configuration)


if __name__ == "__main__":
    main()
