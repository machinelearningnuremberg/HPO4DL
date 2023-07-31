from typing import List, Dict
from pathlib import Path
import glob
import os

from timm_scripts import train


def objective_function(configuration: Dict, epoch: int, previous_epoch: int, checkpoint_path: Path) -> List:
    seed = 0
    model = 'mobilevit_xxs'
    dataset = 'hfds/cifar10'
    # data_dir = '../hpo4dl/data/cats_vs_dogs_mini'
    dataset_download = True

    path_parts = checkpoint_path.parts
    experiment_name = path_parts[-2]
    output_checkpoint_path = Path(*path_parts[:-2])
    resume_checkpoint_path = checkpoint_path if previous_epoch > 0 else ''

    metric = train.main_with_args(
        dataset=dataset,
        model=model,
        epochs=epoch,
        seed=seed,
        dataset_download=dataset_download,
        resume=str(resume_checkpoint_path),
        output=str(output_checkpoint_path),
        experiment=experiment_name,
        checkpoint_hist=1,
        val_split="test",
        eval_metric='top_1',
        batch_size=128,
        **configuration,
    )

    # remove extra checkpoint files
    trial_checkpoint_path = Path(*path_parts[:-1]) / 'checkpoint*'
    files = glob.glob(str(trial_checkpoint_path))

    for file in files:
        try:
            os.remove(file)
        except OSError as e:
            print("Error: %s : %s" % (file, e.strerror))

    return metric
