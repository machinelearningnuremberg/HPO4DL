import sys
import os
import random
import ConfigSpace as CS
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        prog='hpo4dl',
        description='Hyperparameter optimization using hpo4dl and timm libraries',
    )
    parser.add_argument('--output-dir', default='./hpo4dl_results',
                        help='path to save results. (default: ./hpo4dl_results)')
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
    parser.add_argument('--optimizer', type=str, default='hyperband',
                        help='optimizer name (default: dyhpo)')
    parser.add_argument('--max-epochs', type=int, default=27,
                        help='maximum epochs (default: 27)')
    parser.add_argument('--max-budget', type=int, default=1000,
                        help='maximum budget in epochs (default: 1000)')
    args = parser.parse_args()

    sys.argv = ['']

    from dummy_objective import DummyObjective
    from timm_objective import TimmObjective

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

    storage_dir = os.path.expanduser(args.storage_dir)
    objective_instance = TimmObjective(
        dataset=args.dataset,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        storage_dir=storage_dir,
    )
    # objective_instance = DummyObjective(seed=args.seed)

    tuner = Tuner(
        objective_function=objective_instance.objective_function,
        configuration_space=objective_instance.configspace,
        minimize=False,
        max_budget=args.max_budget,
        optimizer=args.optimizer,
        seed=args.seed,
        max_epochs=args.max_epochs,
        output_path=args.output_dir,
        storage_path=storage_dir,
    )

    with open(tuner.output_path / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    incumbent = tuner.run()
    print("Incumbent Info", incumbent)


if __name__ == "__main__":
    main()
