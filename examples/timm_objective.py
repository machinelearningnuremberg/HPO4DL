from typing import List, Dict
from pathlib import Path
import glob
import os

import torch.cuda
import ConfigSpace as CS

from timm_scripts import train


class TimmObjective:
    def __init__(self, dataset='torch/cifar10', train_split='train', val_split='validation', seed=42, storage_dir=None):
        self.dataset = dataset
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        if storage_dir is None:
            storage_dir = '~/hpo4dl/data'
        self.storage_dir = Path(storage_dir)

        # self.dataset = 'torch/inaturalist'
        # self.data_dir = './data/inaturalist'
        # self.train_split = "kingdom/train"
        # self.val_split = "kingdom/validation"

    @property
    def configspace(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace(seed=self.seed)
        config_space.add_hyperparameters([
            CS.UniformFloatHyperparameter('lr', lower=1e-5, upper=1, log=True),
            CS.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1, log=True),
            CS.CategoricalHyperparameter('model', choices=["mobilevit_xxs", "dla60x_c", "edgenext_xx_small"]),
            CS.CategoricalHyperparameter('opt', choices=["sgd", "adam"]),
            CS.UniformFloatHyperparameter('momentum', lower=0.1, upper=0.99),
        ])
        cond = CS.EqualsCondition(config_space['momentum'], config_space['opt'], "sgd")
        config_space.add_condition(cond)
        return config_space

    def objective_function(self, configuration: Dict, epoch: int, previous_epoch: int, checkpoint_path: str) -> List:
        data_dir = str(self.storage_dir / 'data' / self.dataset)
        dataset_download = True
        use_amp = torch.cuda.is_available()

        checkpoint_path = Path(checkpoint_path)
        path_parts = checkpoint_path.parts
        experiment_name = path_parts[-2]
        output_checkpoint_path = Path(*path_parts[:-2])
        resume_checkpoint_path = checkpoint_path if previous_epoch > 0 else ''

        evaluated_metrics = train.main_with_args(
            dataset=self.dataset,
            data_dir=data_dir,
            epochs=epoch,
            seed=self.seed,
            dataset_download=dataset_download,
            resume=str(resume_checkpoint_path),
            output=str(output_checkpoint_path),
            experiment=experiment_name,
            checkpoint_hist=1,
            train_split=self.train_split,
            val_split=self.val_split,
            eval_metric='top1',
            batch_size=128,
            sched="None",
            workers=4,
            amp=use_amp,
            **configuration,
        )

        # Normalize top-1 accuracy metric
        for item in evaluated_metrics:
            item['metric'] /= 100

        # Renaming 'last.pth.tar' to 'last'
        old_name = Path(*path_parts[:-1]) / 'last.pth.tar'
        new_name = Path(*path_parts[:-1]) / path_parts[-1]

        try:
            os.rename(str(old_name), str(new_name))
        except OSError as ex:
            print(f"Error renaming {old_name} to {new_name}: {ex.strerror}")

        # remove extra checkpoint files made by timm script to save space.
        trial_checkpoint_path = Path(*path_parts[:-1]) / 'checkpoint*'
        files = glob.glob(str(trial_checkpoint_path))

        for file in files:
            try:
                os.remove(file)
            except OSError as ex:
                print(f"Error: {file} : {ex.strerror}")

        # return metric should be a dictionary with 'epoch' and 'metric'.
        return evaluated_metrics
